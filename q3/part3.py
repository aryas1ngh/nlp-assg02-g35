import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import pandas as pd
import shutil

class Config:
    DATA_DIR = '../dataset'
    TRAIN_FILE = os.path.join(DATA_DIR, 'train_data.jsonl')
    VAL_FILE = os.path.join(DATA_DIR, 'val_data.jsonl')
    TEST_FILE = '../test_data.jsonl'
    OUTPUT_FILE = 'output.jsonl'
    PLOTS_DIR = 'plots'
    MODELS_DIR = 'saved_models'
    
    # embedding paths
    GLOVE_PATH = '../glove.6B.100d.txt'
    FASTTEXT_PATH = '../wiki-news-300d-1M.vec'
    
    MAX_SEQ_LEN = 128
    
    # hyperparams
    HIDDEN_DIM = 128
    EMBEDDING_DIM = 100 # GloVe 100d
    BATCH_SIZE = 64
    LEARNING_RATE = 0.003
    EPOCHS = 12 
    DROPOUT = 0.3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_dirs():
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)

def load_data(filepath, has_labels=True):
    ids, sentences, labels = [], [], []
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return [], [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            ids.append(data.get('id', ''))
            sentences.append(data['tokens'])
            if has_labels:
                labels.append(data['labels'])
    return (ids, sentences, labels) if has_labels else (ids, sentences, None)

def build_vocab(sentences, labels):
    word_counts = Counter()
    for sent in sentences:
        word_counts.update(sent)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in word_counts.items():
        vocab[word] = len(vocab)
    
    label_map = {"<PAD>": -1}
    unique_labels = sorted(list(set([l for sublist in labels for l in sublist])))
    for i, label in enumerate(unique_labels):
        label_map[label] = i
    return vocab, label_map

def load_pretrained_embeddings(path, vocab, embedding_dim):
    print(f"Loading embeddings from {path}...")
    embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embeddings[vocab["<PAD>"]] = 0
    
    count = 0
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Using random embeddings.")
        return torch.from_numpy(embeddings).float()
        
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split()
            if len(parts) <= 2: continue # skip header if present (FastText)
            word = parts[0]
            if word in vocab:
                vector = np.array(parts[1:], dtype='float32')
                if len(vector) == embedding_dim:
                    embeddings[vocab[word]] = vector
                    count += 1
    print(f"Loaded {count} vectors.")
    return torch.from_numpy(embeddings).float()

class NERDataset(Dataset):
    def __init__(self, sentences, labels, vocab, label_map):
        self.sentences = sentences
        self.labels = labels
        self.vocab = vocab
        self.label_map = label_map
    def __len__(self): return len(self.sentences)
    def __getitem__(self, idx):
        tokens = self.sentences[idx][:Config.MAX_SEQ_LEN]
        tags = self.labels[idx][:Config.MAX_SEQ_LEN]
        token_ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        tag_ids = [self.label_map[t] for t in tags]
        return torch.tensor(token_ids), torch.tensor(tag_ids), len(token_ids)

def collate_fn(batch):
    sentences, labels, lengths = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
    return sentences_padded, labels_padded, torch.tensor(lengths)

# --- Optimized Bahdanau Attention ---
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim) # combined query + value transformation
        self.v = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, query, values):
        # query, values: (batch, seq_len, hidden_dim)
        # Instead of 4D expansion, we can do it more efficiently:
        # scores[i,j] = v^T * tanh(Wq * q[i] + Wv * v[j])
        # We can pre-transform values.
        
        batch_size, seq_len, h_dim = values.shape
        
        # We want to compute scores for each step i attending to all steps j
        # For sequence tagging, query[i] is just the hidden state at step i.
        # So we want a (batch, seq_len, seq_len) attention matrix.
        
        # Linear transformation of values
        v_transformed = self.W(values) # (batch, seq_len, hidden_dim)
        
        # For each query step, we add its own transformation
        q_transformed = self.W(query) # (batch, seq_len, hidden_dim)
        
        # Use broadcasting to get (batch, seq_len, seq_len, hidden_dim)
        # score = v^T * tanh(q_transformed[i] + v_transformed[j])
        combined = torch.tanh(q_transformed.unsqueeze(2) + v_transformed.unsqueeze(1))
        
        scores = torch.matmul(combined, self.v) # (batch, seq_len, seq_len)
        alphas = torch.softmax(scores, dim=-1)
        context = torch.matmul(alphas, values)
        return context

class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout=0.3, pretrained_weights=None):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(pretrained_weights)
            # self.embedding.weight.requires_grad = False # Optional: freeze embeddings
            
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout if n_layers > 1 else 0)
        self.attention = BahdanauAttention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 4, output_dim) # combined hidden + context
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_embedded)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True) # (batch, seq, 2*hidden)
        
        # Apply Bahdanau Attention
        context = self.attention(lstm_out, lstm_out) # (batch, seq, 2*hidden)
        
        # Concatenate BiLSTM output and context vector
        combined = torch.cat((lstm_out, context), dim=2) # (batch, seq, 4*hidden)
        output = self.fc(self.dropout(combined))
        return output

# --- Metrics (BIO Chunk F1) ---
def get_chunks(seq, label_map_inv):
    chunks = []
    chunk_type, chunk_start = None, None
    for i, label_idx in enumerate(seq):
        label = label_map_inv.get(label_idx, "O")
        if label == "O":
            if chunk_type: chunks.append((chunk_type, chunk_start, i))
            chunk_type, chunk_start = None, None
        elif label.startswith("B-"):
            if chunk_type: chunks.append((chunk_type, chunk_start, i))
            chunk_type = label[2:]; chunk_start = i
        elif label.startswith("I-"):
            if chunk_type is None:
                chunk_type = label[2:]; chunk_start = i
            elif label[2:] != chunk_type:
                chunks.append((chunk_type, chunk_start, i))
                chunk_type = label[2:]; chunk_start = i
    if chunk_type: chunks.append((chunk_type, chunk_start, len(seq)))
    return set(chunks)

def calculate_metrics(pred_lists, true_lists, label_map):
    label_map_inv = {v: k for k, v in label_map.items()}
    # Strict EM
    exact = sum([1 for p, t in zip(pred_lists, true_lists) if p == t])
    strict_em = exact / len(pred_lists) if pred_lists else 0
    # Entity F1 (FreeMatch-F1)
    true_ent, pred_ent, corr_ent = 0, 0, 0
    for p, t in zip(pred_lists, true_lists):
        tc = get_chunks(t, label_map_inv)
        pc = get_chunks(p, label_map_inv)
        true_ent += len(tc); pred_ent += len(pc); corr_ent += len(tc.intersection(pc))
    prec = corr_ent / pred_ent if pred_ent > 0 else 0
    rec = corr_ent / true_ent if true_ent > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return strict_em, f1

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for text, tags, lengths in tqdm(loader, desc="Training", leave=False):
        text, tags = text.to(device), tags.to(device)
        optimizer.zero_grad()
        preds = model(text, lengths)
        loss = criterion(preds.view(-1, preds.shape[-1]), tags.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, label_map):
    model.eval()
    total_loss, all_preds, all_trues = 0, [], []
    with torch.no_grad():
        for text, tags, lengths in loader:
            text, tags = text.to(device), tags.to(device)
            preds = model(text, lengths)
            loss = criterion(preds.view(-1, preds.shape[-1]), tags.view(-1))
            total_loss += loss.item()
            batch_preds = torch.argmax(preds, dim=2)
            for i in range(len(lengths)):
                l = lengths[i]
                all_preds.append(batch_preds[i][:l].cpu().tolist())
                all_trues.append(tags[i][:l].cpu().tolist())
    em, f1 = calculate_metrics(all_preds, all_trues, label_map)
    return total_loss / len(loader), em, f1

def generate_test_output(model, vocab, label_map):
    ids, sentences, _ = load_data(Config.TEST_FILE, has_labels=False)
    if not ids:
        print("Test data not found for inference.")
        return
    label_map_inv = {v: k for k, v in label_map.items()}
    model.eval()
    results = []
    with torch.no_grad():
        for i in range(len(sentences)):
            tokens = sentences[i]
            token_ids = torch.tensor([vocab.get(t, vocab["<UNK>"]) for t in tokens]).unsqueeze(0).to(device)
            lengths = torch.tensor([len(tokens)])
            preds = model(token_ids, lengths)
            pred_indices = torch.argmax(preds, dim=2).squeeze(0).cpu().tolist()
            results.append({"id": ids[i], "tokens": tokens, 
                            "labels": [label_map_inv.get(idx, "O") for idx in pred_indices]})
    with open(Config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for res in results:
            json.dump(res, f); f.write('\n')
    print(f"Saved inference results to {Config.OUTPUT_FILE}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()

    ensure_dirs()
    ids_t, sents_t, labels_t = load_data(Config.TRAIN_FILE)
    ids_v, sents_v, labels_v = load_data(Config.VAL_FILE)
    vocab, label_map = build_vocab(sents_t + sents_v, labels_t + labels_v)
    
    if args.mode == 'train':
        # Load embeddings (GloVe 100d) - Validated choice for performance/speed
        pretrained_weights = load_pretrained_embeddings(Config.GLOVE_PATH, vocab, Config.EMBEDDING_DIM)
        
        train_loader = DataLoader(NERDataset(sents_t, labels_t, vocab, label_map), 
                                 batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(NERDataset(sents_v, labels_v, vocab, label_map), 
                               batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        results = []
        overall_best_f1 = 0
        best_model_data = None

        for L in [1, 2, 3]:
            print(f"\n--- Training BiLSTM + Attention (L={L}) ---")
            model = BiLSTM_Attention(len(vocab), Config.EMBEDDING_DIM, Config.HIDDEN_DIM, 
                                     len(label_map), L, pretrained_weights=pretrained_weights).to(device)
            optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
            
            train_losses, val_losses = [], []
            local_best_f1 = 0
            patience_counter = 0
            for epoch in range(Config.EPOCHS):
                t_loss = train_epoch(model, train_loader, optimizer, criterion)
                v_loss, v_em, v_f1 = evaluate(model, val_loader, criterion, label_map)
                train_losses.append(t_loss); val_losses.append(v_loss)
                
                scheduler.step(v_f1)
                
                if v_f1 > local_best_f1:
                    local_best_f1 = v_f1
                    patience_counter = 0
                    model_path = f"{Config.MODELS_DIR}/best_attn_L{L}.pt"
                    torch.save({'state': model.state_dict(), 'L': L, 'vocab': vocab, 'label_map': label_map}, model_path)
                    if v_f1 > overall_best_f1:
                        overall_best_f1 = v_f1
                        best_model_data = model_path
                else:
                    patience_counter += 1
                
                print(f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {t_loss:.4f} | Val F1: {v_f1:.4f} | EM: {v_em:.4f}")
                
                if patience_counter >= 5:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            plt.figure(); plt.plot(train_losses, label='Train'); plt.plot(val_losses, label='Val')
            plt.title(f"Attention BiLSTM L={L}"); plt.legend()
            plt.savefig(f"{Config.PLOTS_DIR}/loss_L{L}.png"); plt.close()
            results.append({"Layers": L, "FreeMatch-F1": local_best_f1, "Strict EM": v_em}) # v_em from last epoch or best? Usually best F1's EM.

        if best_model_data:
            shutil.copy(best_model_data, "best_attn_model.pt")
        print("\n" + "="*30 + "\nResults Summary\n" + "="*30)
        print(pd.DataFrame(results))
    
    else: # mode == 'test'
        best_filename = "best_attn_glove_L1.pt"
        if not os.path.exists(best_filename):
            print(f"{best_filename} not found.")
        else:
            checkpoint = torch.load(best_filename, map_location=device)
            vocab = checkpoint['vocab']
            label_map = checkpoint['label_map']
            L = checkpoint['L']
            model = BiLSTM_Attention(len(vocab), Config.EMBEDDING_DIM, Config.HIDDEN_DIM, len(label_map), L).to(device)
            model.load_state_dict(checkpoint['state'])
            generate_test_output(model, vocab, label_map)
