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
    GLOVE_PATH = '../glove.6B.100d.txt' 
    FASTTEXT_PATH = '../wiki-news-300d-1M-subword.vec'    # now using subword embeddings
    
    # hyperparams
    HIDDEN_DIM = 64
    EMBEDDING_DIM_GLOVE = 100
    EMBEDDING_DIM_FASTTEXT = 300
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    EPOCHS = 15
    DROPOUT = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_dirs():
    if not os.path.exists(Config.DATA_DIR):
        print(f"Error: Directory '{Config.DATA_DIR}' not found. Please create it.")
        exit()
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)

# load data from jsonl file
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

# build vocab from sentences and labels
# keep minimum frequency as 1, add <PAD> and <UNK> tokens
def build_vocab(sentences, labels, min_freq=1):
    word_counts = Counter()
    for sent in sentences:
        word_counts.update(sent)
        
    # add <PAD> and <UNK> tokens, then add words having atleast minimum frequency
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
            
    # build label map
    # add <PAD> token, then add labels sorted
    label_map = {"<PAD>": -1} 
    unique_labels = sorted(list(set([l for sublist in labels for l in sublist])))
    for i, label in enumerate(unique_labels):
        label_map[label] = i
        
    return vocab, label_map

# building the torch compatible dataset
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
        token_ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        tag_ids = [self.label_map[t] for t in tags]
        return torch.tensor(token_ids), torch.tensor(tag_ids), len(token_ids)

# standard collate function to pad sequences
def collate_fn(batch):
    sentences, labels, lengths = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
    return sentences_padded, labels_padded, torch.tensor(lengths)


# get B-I-O chunks from label ids
def get_chunks(seq, label_map_inv):
    chunks = []
    chunk_type, chunk_start = None, None
    for i, label_idx in enumerate(seq):
        label = label_map_inv.get(label_idx, "O")
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
        # inside
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

# helper to calculate metrics: exact match and entity f1
def calculate_metrics(pred_lists, true_lists, label_map):
    label_map_inv = {v: k for k, v in label_map.items()}
    # strict EM (exact match)
    exact = sum([1 for p, t in zip(pred_lists, true_lists) if p == t])
    strict_em = exact / len(pred_lists) if pred_lists else 0

    # entity f1 score
    # first get chunks from true and predicted labels
    true_ent, pred_ent, corr_ent = 0, 0, 0
    for p, t in zip(pred_lists, true_lists):
        tc = get_chunks(t, label_map_inv)
        pc = get_chunks(p, label_map_inv)
        true_ent += len(tc)
        pred_ent += len(pc)
        corr_ent += len(tc.intersection(pc))

    # calculate precision and recall
    prec = corr_ent / pred_ent if pred_ent > 0 else 0
    rec = corr_ent / true_ent if true_ent > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return strict_em, f1


# loading glove and fasttext embeddings, as required
# initialize with random normal distribution, if not found
def load_pretrained_embeddings(path, vocab, embed_dim):
    print(f"Loading embeddings from {path}...")

    # failsafe, in case no pretrained embeddings are found
    embedding_matrix = np.random.normal(scale=0.6, size=(len(vocab), embed_dim))
    embedding_matrix[0] = np.zeros(embed_dim)  # important for padding
    
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Using random.")
        return torch.tensor(embedding_matrix, dtype=torch.float32)

    # read file and update matrix
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < embed_dim: continue
            word = parts[0]
            vector = parts[1:]
            if len(vector) != embed_dim: continue
            if word in vocab:
                embedding_matrix[vocab[word]] = np.array(vector, dtype=np.float32)
    return torch.tensor(embedding_matrix, dtype=torch.float32)

################## main model logic #####################
# architecture: 
# input -> embedding -> dropout -> (bi)lstm -> dropout -> fully connected -> output
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, pretrained_embeddings=None, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # add dropout only if more than 1 layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim*2, output_dim) 
        self.dropout = nn.Dropout(dropout)
    
    # forward pass
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_embedded)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return self.fc(output)


# train loop
# using tqdm for progress bar
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for text, tags, lengths in pbar:
        text, tags = text.to(device), tags.to(device)
        optimizer.zero_grad()
        preds = model(text, lengths)
        loss = criterion(preds.view(-1, preds.shape[-1]), tags.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    return total_loss / len(loader)

# calculate loss, backprop, and update weights
def evaluate(model, loader, criterion, label_map):
    model.eval()
    total_loss = 0
    all_preds, all_trues = [], []
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
    strict_em, f1 = calculate_metrics(all_preds, all_trues, label_map)
    return total_loss / len(loader), strict_em, f1

# generate the output of inference
def generate_test_output(model, vocab, label_map):
    # load test data
    ids, sentences, _ = load_data(Config.TEST_FILE, has_labels=False)
    if not ids:
        print("No test data found.")
        return
    print(f"Predicting on {len(sentences)} test sentences...")

    # invert the label map
    label_map_inv = {v: k for k, v in label_map.items()}
    model.eval()
    results = []
    # inference loop
    with torch.no_grad():
        for i in tqdm(range(len(sentences)), desc="Inference"):
            tokens = sentences[i]
            # convert tokens to ids
            token_ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
            # convert to tensor
            tensor_ids = torch.LongTensor(token_ids).unsqueeze(0).to(device)
            length = torch.tensor([len(token_ids)])
            # get predictions
            preds = model(tensor_ids, length)
            pred_indices = torch.argmax(preds, dim=2).squeeze(0).cpu().tolist()
            pred_labels = [label_map_inv[idx] for idx in pred_indices]

            results.append({"id": ids[i], "tokens": tokens, "labels": pred_labels})
            
    with open(Config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for res in results:
            json.dump(res, f)
            f.write('\n')
    print(f"Saved to {Config.OUTPUT_FILE}")


# driver
if __name__ == "__main__":
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['test', 'ablate'], help='Mode')
    parser.add_argument('--pt', type=str, required=False, choices=['glove', 'fasttext'], help='Embedding type')
    parser.add_argument('--layers', type=int, required=False, default=2, help='Num LSTM layers')
    args = parser.parse_args()

    # inference mode
    if args.mode == 'test':
        if not os.path.exists('best_model.pt'):
            print("Error: best_model.pt not found. Run train or ablate first.")
            exit()
        
        # load best model
        print("Loading best_model.pt...")
        checkpoint = torch.load('best_model.pt', map_location=device)
        vocab = checkpoint['vocab']
        label_map = checkpoint['label_map']
        n_layers = checkpoint['layers']
        emb_type = checkpoint['embedding_type']
        
        # load weights according to the embedding type
        if emb_type == 'glove':
            emb_dim = Config.EMBEDDING_DIM_GLOVE
            emb_path = Config.GLOVE_PATH
        else:
            emb_dim = Config.EMBEDDING_DIM_FASTTEXT
            emb_path = Config.FASTTEXT_PATH
            
        pretrained_weights = load_pretrained_embeddings(emb_path, vocab, emb_dim)
        model = LSTMModel(len(vocab), emb_dim, Config.HIDDEN_DIM, len(label_map), 
                          n_layers, pretrained_weights, Config.DROPOUT).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        generate_test_output(model, vocab, label_map)
        exit()

    # load data for ablation study
    print(f"Loading data from {Config.DATA_DIR}...")
    train_ids, train_sents, train_labels = load_data(Config.TRAIN_FILE)
    val_ids, val_sents, val_labels = load_data(Config.VAL_FILE)
    
    if not train_sents or not val_sents:
        print("Error: Train/Val data missing.")
        exit()

    vocab, label_map = build_vocab(train_sents + val_sents, train_labels + val_labels)
    print(f"Vocab: {len(vocab)} | Labels: {len(label_map)}")

    train_loader = DataLoader(NERDataset(train_sents, train_labels, vocab, label_map), 
                              batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(NERDataset(val_sents, val_labels, vocab, label_map), 
                            batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # ablation study
    if args.mode == 'ablate':
        results = []
        overall_best_f1 = 0.0
        best_model_filename = ""
        
        print("\nSTARTING ABLATION STUDY of 6 experiments...")
        # load weights according to the embedding type
        for pt in ['glove', 'fasttext']:
            if pt == 'glove':
                e_dim = Config.EMBEDDING_DIM_GLOVE
                e_path = Config.GLOVE_PATH
            else:
                e_dim = Config.EMBEDDING_DIM_FASTTEXT
                e_path = Config.FASTTEXT_PATH
            
            emb_weights = load_pretrained_embeddings(e_path, vocab, e_dim)

            for L in [1,2,3]:
                print(f"\nEmbedding: {pt} | Layers: {L}")
                print("____"*20)
                model = LSTMModel(len(vocab), e_dim, Config.HIDDEN_DIM, len(label_map), L, emb_weights, Config.DROPOUT).to(device)
                optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
                
                train_losses, val_losses = [], []
                local_best_f1 = 0.0
                local_best_em = 0.0
                
                # train this specific model config
                for epoch in range(Config.EPOCHS):
                    t_loss = train_epoch(model, train_loader, optimizer, criterion)
                    v_loss, v_em, v_f1 = evaluate(model, val_loader, criterion, label_map)
                    
                    train_losses.append(t_loss)
                    val_losses.append(v_loss)
                    
                    # track local best for this specific model config
                    if v_f1 > local_best_f1:
                        local_best_f1 = v_f1
                        local_best_em = v_em
                        # save the best version of this model
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

                    print(f"Epoch: {epoch+1}/{Config.EPOCHS} | train_loss: {t_loss:.4f} | val_loss: {v_loss:.4f} | val_f1: {v_f1:.4f} | strict_em: {v_em:.4f}")

                # experiment finished. record results
                results.append({'Embedding': pt, 'Layers': L, 'F1': local_best_f1, 'EM': local_best_em})
                
                # check if this model is the new global champion
                if local_best_f1 > overall_best_f1:
                    overall_best_f1 = local_best_f1
                    best_model_filename = f"model_{pt}_L{L}.pt"
                    print(f"  >> new global best found! ({pt}, L={L}, F1={local_best_f1:.4f})")

                # Plot
                plt.figure()
                plt.plot(train_losses, label='Train')
                plt.plot(val_losses, label='Val')
                plt.title(f"{pt} - {L} Layers")
                plt.legend()
                plt.savefig(f"{Config.PLOTS_DIR}/loss_{pt}_L{L}.png")
                plt.close()

        # end of all experiments
        # save the best model to 'best_model.pt'
        print("\n" + "="*50)
        print(f"Overall Best Model: {best_model_filename} (F1: {overall_best_f1:.4f})")
        if best_model_filename:
            src = os.path.join(Config.MODELS_DIR, best_model_filename)
            shutil.copy(src, 'best_model.pt')
            print("Copied best model to 'best_model.pt'")
        
        # print hyperparameters
        print("Hyperparameters:")
        print(f"  Hidden Dim:     {Config.HIDDEN_DIM}")
        print(f"  Batch Size:     {Config.BATCH_SIZE}")
        print(f"  Learning Rate:  {Config.LEARNING_RATE}")
        print(f"  Weight Decay:   {Config.WEIGHT_DECAY}")
        print(f"  Dropout:        {Config.DROPOUT}")
        print(f"  Max Epochs:     {Config.EPOCHS}")

        # print the results table
        res_df = pd.DataFrame(results)
        print(res_df)