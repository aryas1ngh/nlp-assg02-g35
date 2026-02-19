# data parsing
import json
from collections import Counter
# model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from torchcrf import CRF
# plots
import matplotlib.pyplot as plt
# terminal handling
import os
import argparse
import shutil
from tqdm import tqdm
import pandas as pd

class Config:
    # data paths
    DATA_DIR = '../dataset'
    TRAIN_FILE = os.path.join(DATA_DIR, 'train_data.jsonl')
    VAL_FILE = os.path.join(DATA_DIR, 'val_data.jsonl')
    TEST_FILE = '../test_data.jsonl'
    OUTPUT_FILE = 'tagged_output.jsonl'
    PLOTS_DIR = 'plots'
    MODELS_DIR = 'saved_models'
    
    # embedding paths
    # GLOVE_PATH = '../glove.6B.100d.txt' 
    FASTTEXT_PATH = '../wiki-news-300d-1M-subword.vec'
    
    # EMBEDDING_DIM_GLOVE = 100
    EMBEDDING_DIM_FASTTEXT = 300
    
    HIDDEN_DIM = 128
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0015
    WEIGHT_DECAY = 3e-4
    EPOCHS = 15
    DROPOUT = 0.5
    FREEZE_EMBEDDINGS = True
    LR_PATIENCE = 3
    EARLY_STOP_PATIENCE = 7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_dirs():
    if not os.path.exists(Config.DATA_DIR):
        print(f"Error: Directory '{Config.DATA_DIR}' not found. Please create it.")
        exit()
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)

def load_data(filepath, has_labels=True):
    ids, sentences, labels = [], [], []
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return [], [], []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                ids.append(data.get('id', ''))
                sentences.append(data['tokens'])
                if has_labels:
                    labels.append(data['labels'])
            except json.JSONDecodeError:
                continue
    return (ids, sentences, labels) if has_labels else (ids, sentences, None)

# better vocabulary building with min_freq=2 for noise reduction
def build_vocab(sentences, labels, min_freq=2):
    word_counts = Counter()
    for sent in sentences:
        word_counts.update([w.lower() for w in sent])
        
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    # use 0 for pad instead of -1 (crf compatible)
    label_map = {"<PAD>": 0}
    unique_labels = sorted(list(set([l for sublist in labels for l in sublist])))
    for i, label in enumerate(unique_labels, start=1):
        label_map[label] = i
        
    return vocab, label_map

class NERDataset(Dataset):
    def __init__(self, sentences, labels, vocab, label_map):
        self.sentences = sentences
        self.labels = labels
        self.vocab = vocab
        self.label_map = label_map
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        tags = self.labels[idx]
        token_ids = [self.vocab.get(t.lower(), self.vocab["<UNK>"]) for t in tokens]
        tag_ids = [self.label_map[t] for t in tags]
        return torch.tensor(token_ids), torch.tensor(tag_ids), len(token_ids)

def collate_fn(batch):
    sentences, labels, lengths = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)  # 0 instead of -1
    return sentences_padded, labels_padded, torch.tensor(lengths)

def get_chunks(seq, label_map_inv):
    chunks = []
    chunk_type, chunk_start = None, None
    for i, label_idx in enumerate(seq):
        label = label_map_inv.get(label_idx, "O")
        # skip padding labels
        if label == "<PAD>":
            continue

        # other
        if label == "O":
            if chunk_type:
                chunks.append((chunk_type, chunk_start, i))
                chunk_type, chunk_start = None, None

        # begin
        elif label.startswith("B-"):
            if chunk_type:
                chunks.append((chunk_type, chunk_start, i))
            chunk_type = label[2:]
            chunk_start = i

        # intermediate
        elif label.startswith("I-"):
            if chunk_type is None:
                chunk_type = label[2:]
                chunk_start = i
            elif label[2:] != chunk_type:
                chunks.append((chunk_type, chunk_start, i))
                chunk_type = label[2:]
                chunk_start = i
    if chunk_type:
        chunks.append((chunk_type, chunk_start, len(seq)))
    return set(chunks)

# get f1, strict em
def calculate_metrics(pred_lists, true_lists, label_map):
    label_map_inv = {v: k for k, v in label_map.items()}
    exact = sum([1 for p, t in zip(pred_lists, true_lists) if p == t])
    strict_em = exact / len(pred_lists) if pred_lists else 0

    true_ent, pred_ent, corr_ent = 0, 0, 0
    for p, t in zip(pred_lists, true_lists):
        tc = get_chunks(t, label_map_inv)
        pc = get_chunks(p, label_map_inv)
        true_ent += len(tc)
        pred_ent += len(pc)
        corr_ent += len(tc.intersection(pc))

    # get precision, recall --> f1
    prec = corr_ent / pred_ent if pred_ent > 0 else 0
    rec = corr_ent / true_ent if true_ent > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return strict_em, f1

# load embeddings
def load_pretrained_embeddings(path, vocab, embed_dim):
    print(f"Loading embeddings from {path}...")

    # failsafe, in case no pretrained embeddings are found
    embedding_matrix = np.random.normal(scale=0.6, size=(len(vocab), embed_dim))
    embedding_matrix[0] = np.zeros(embed_dim)
    
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Using random.")
        return torch.tensor(embedding_matrix, dtype=torch.float32)

    covered = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.rstrip().split()
            # correct dimension check (word + embed_dim values)
            if len(parts) < embed_dim + 1:
                continue
            word = parts[0].lower()
            # take exactly embed_dim values
            vector = parts[1:embed_dim + 1]
            if len(vector) != embed_dim:
                continue
            try:
                if word in vocab:
                    embedding_matrix[vocab[word]] = np.array(vector, dtype=np.float32)
                    covered += 1
            except (ValueError, IndexError):
                continue
    
    print(f"Embedding coverage: {covered}/{len(vocab)} ({100*covered/len(vocab):.1f}%)")
    return torch.tensor(embedding_matrix, dtype=torch.float32)

# improved model with layernorm and proper initialization
class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 pretrained_embeddings=None, dropout=0.5, freeze_embeddings=True):
        super(BiLSTMCRF, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # freeze embeddings to prevent overfitting
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        
        # add layernorm for better gradient flow
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.crf = CRF(output_dim, batch_first=True)
        
        # proper weight initialization
        self._init_weights()
    
    # added wt initialization
    def _init_weights(self):
        # initialize LSTM weights according to their types
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        # initialize FC layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, text, text_lengths, labels=None, mask=None):
        embedded = self.embedding(text)
        embedded = self.layer_norm(embedded)  # Better gradient flow
        embedded = self.dropout(embedded)
        
        packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_embedded)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.dropout(output)
        
        emissions = self.fc(output)
        
        if labels is not None:
            return -self.crf(emissions, labels, mask=mask, reduction='mean')
        else:
            return self.crf.decode(emissions, mask=mask)

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for text, tags, lengths in pbar:
        text, tags = text.to(device), tags.to(device)
        
        # efficient mask creation (no loop)
        mask = (text != 0)
        
        optimizer.zero_grad()
        loss = model(text, lengths, labels=tags, mask=mask)
        loss.backward()
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)

def evaluate(model, loader, label_map):
    model.eval()
    total_loss = 0
    all_preds, all_trues = [], []
    
    with torch.no_grad():
        for text, tags, lengths in loader:
            text, tags = text.to(device), tags.to(device)
            mask = (text != 0)
            
            loss = model(text, lengths, labels=tags, mask=mask)
            total_loss += loss.item()
            
            batch_preds = model(text, lengths, mask=mask)
            
            for i in range(len(lengths)):
                l = lengths[i]
                all_preds.append(batch_preds[i][:l])
                all_trues.append(tags[i][:l].cpu().tolist())
    
    strict_em, f1 = calculate_metrics(all_preds, all_trues, label_map)
    return total_loss / len(loader), strict_em, f1


# generate tags for test data
def generate_test_output(model, vocab, label_map):
    ids, sentences, _ = load_data(Config.TEST_FILE, has_labels=False)
    if not ids:
        print("No test data found.")
        return
    print(f"Predicting on {len(sentences)} test sentences...")

    label_map_inv = {v: k for k, v in label_map.items()}
    model.eval()
    results = []
    
    with torch.no_grad():
        for i in tqdm(range(len(sentences)), desc="Inference"):
            tokens = sentences[i]
            token_ids = [vocab.get(t.lower(), vocab["<UNK>"]) for t in tokens]
            tensor_ids = torch.LongTensor(token_ids).unsqueeze(0).to(device)
            length = torch.tensor([len(token_ids)])
            mask = torch.ones((1, len(token_ids)), dtype=torch.bool, device=device)
            
            pred_indices = model(tensor_ids, length, mask=mask)[0]
            pred_labels = [label_map_inv[idx] for idx in pred_indices]

            results.append({"id": ids[i], "tokens": tokens, "labels": pred_labels})
            
    with open(Config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for res in results:
            json.dump(res, f)
            f.write('\n')
    print(f"Saved to {Config.OUTPUT_FILE}")

if __name__ == "__main__":
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['test', 'ablate'], help='Mode')
    parser.add_argument('--pt', type=str, required=False, choices=['glove', 'fasttext'], help='Embedding type')
    parser.add_argument('--layers', type=int, required=False, default=2, help='Num LSTM layers')
    args = parser.parse_args()

    if args.mode == 'test':
        if not os.path.exists('best_model.pt'):
            print("Error: best_model.pt not found. Run ablate first.")
            exit()
        
        print("Loading best_model.pt...")
        checkpoint = torch.load('best_model.pt', map_location=device)
        vocab = checkpoint['vocab']
        label_map = checkpoint['label_map']
        n_layers = checkpoint['layers']
        emb_type = checkpoint['embedding_type']
        
        if emb_type == 'glove':
            emb_dim = Config.EMBEDDING_DIM_GLOVE
            emb_path = Config.GLOVE_PATH
        else:
            emb_dim = Config.EMBEDDING_DIM_FASTTEXT
            emb_path = Config.FASTTEXT_PATH
            
        pretrained_weights = load_pretrained_embeddings(emb_path, vocab, emb_dim)
        model = BiLSTMCRF(len(vocab), emb_dim, Config.HIDDEN_DIM, len(label_map), n_layers, pretrained_weights, Config.DROPOUT, Config.FREEZE_EMBEDDINGS).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        generate_test_output(model, vocab, label_map)
        exit()

    # Load data
    print(f"Loading data from {Config.DATA_DIR}...")
    train_ids, train_sents, train_labels = load_data(Config.TRAIN_FILE)
    val_ids, val_sents, val_labels = load_data(Config.VAL_FILE)
    
    if not train_sents or not val_sents:
        print("Error: Train/Val data missing.")
        exit()

    vocab, label_map = build_vocab(train_sents + val_sents, train_labels + val_labels)
    print(f"Vocab: {len(vocab)} | Labels: {len(label_map)}")

    train_loader = DataLoader(NERDataset(train_sents, train_labels, vocab, label_map), batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(NERDataset(val_sents, val_labels, vocab, label_map), batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    if args.mode == 'ablate':
        results = []
        overall_best_f1 = 0.0
        best_model_filename = ""
        
        print("STARTING ABLATION STUDY: 3 experiments (1 embedding, 3 layers)")
        
        # only test fasttext now
        for pt in ['fasttext']:
            if pt == 'glove':
                e_dim = Config.EMBEDDING_DIM_GLOVE
                e_path = Config.GLOVE_PATH
            else:
                e_dim = Config.EMBEDDING_DIM_FASTTEXT
                e_path = Config.FASTTEXT_PATH
            
            emb_weights = load_pretrained_embeddings(e_path, vocab, e_dim)

            for L in [1, 2, 3]:
                print(f"Experiment: {pt.upper()} + {L} Layer(s)")
                
                model = BiLSTMCRF(len(vocab), e_dim, Config.HIDDEN_DIM, len(label_map), L, emb_weights, Config.DROPOUT, Config.FREEZE_EMBEDDINGS).to(device)
                
                optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
                
                # better lr scheduling
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=Config.LR_PATIENCE)
                
                train_losses, val_losses, val_f1s = [], [], []
                local_best_f1 = 0.0
                local_best_em = 0.0
                patience_counter = 0
                
                for epoch in range(Config.EPOCHS):
                    t_loss = train_epoch(model, train_loader, optimizer)
                    v_loss, v_em, v_f1 = evaluate(model, val_loader, label_map)
                    
                    scheduler.step(v_f1)
                    
                    train_losses.append(t_loss)
                    val_losses.append(v_loss)
                    val_f1s.append(v_f1)
                    
                    # track local best
                    if v_f1 > local_best_f1:
                        local_best_f1 = v_f1
                        local_best_em = v_em
                        patience_counter = 0
                        
                        # save best version of this config
                        local_filename = os.path.join(Config.MODELS_DIR, f"model_{pt}_L{L}.pt")
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'vocab': vocab,
                            'label_map': label_map,
                            'embedding_type': pt,
                            'layers': L,
                            'hidden_dim': Config.HIDDEN_DIM,
                            'output_dim': len(label_map)
                        }, local_filename)
                        status = " ==> local best!"
                    else:
                        patience_counter += 1
                        status = ""
                    
                    print(f"Epoch {epoch+1}/{Config.EPOCHS} | train loss: {t_loss:.4f} | val loss: {v_loss:.4f} | val F1: {v_f1:.4f} | val em: {v_em:.4f} {status}")
                    
                    # early stopping
                    if patience_counter >= Config.EARLY_STOP_PATIENCE:
                        print(f"  >> Early stop at epoch {epoch+1} (no improvement for {Config.EARLY_STOP_PATIENCE} epochs)")
                        break
                
                # record results
                results.append({
                    'Embedding': pt, 
                    'Layers': L, 
                    'F1': local_best_f1, 
                    'EM': local_best_em
                })
                
                # check if global best
                if local_best_f1 > overall_best_f1:
                    overall_best_f1 = local_best_f1
                    best_model_filename = f"model_{pt}_L{L}.pt"
                    print(f"\nNEW GLOBAL BEST: {pt} L={L}, F1={local_best_f1:.4f}")
                
                # plot                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # loss subplot
                ax1.plot(train_losses, label='train loss')
                ax1.plot(val_losses, label='val loss')
                ax1.set_title(f'{pt}, {L} layers: Loss')
                ax1.set_xlabel('Epoch')
                ax1.legend()
                
                # f1 subplot
                ax2.plot(val_f1s, label='val F1', color='green')
                ax2.set_title(f'{pt}, {L} layers: F1 Score')
                ax2.set_xlabel('Epoch')
                ax2.legend()
                
                plt.tight_layout()
                plt.savefig(f"{Config.PLOTS_DIR}/metrics_{pt}_L{L}.png")
                plt.close()


        # summary
        print("ABLATION STUDY COMPLETE")
        print(f"\nOverall Best: {best_model_filename} (F1: {overall_best_f1:.4f})")
        
        if best_model_filename:
            src = os.path.join(Config.MODELS_DIR, best_model_filename)
            shutil.copy(src, 'best_model.pt')
            print(f" Copied to 'best_model.pt'\n")
        
        # print config
        print("Hyperparameters:")
        print(f"  Hidden Dim:     {Config.HIDDEN_DIM}")
        print(f"  Batch Size:     {Config.BATCH_SIZE}")
        print(f"  Learning Rate:  {Config.LEARNING_RATE}")
        print(f"  Weight Decay:   {Config.WEIGHT_DECAY}")
        print(f"  Dropout:        {Config.DROPOUT}")
        print(f"  Max Epochs:     {Config.EPOCHS}")
        print(f"  Freeze Emb:     {Config.FREEZE_EMBEDDINGS}\n")
        
        # results table
        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values('F1', ascending=False)
        print("Results Table (sorted by F1):")
        print(res_df.to_string(index=False))