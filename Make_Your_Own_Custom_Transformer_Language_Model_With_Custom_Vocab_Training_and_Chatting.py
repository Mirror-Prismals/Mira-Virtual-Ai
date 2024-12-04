import json
from datetime import datetime
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import re
from tqdm import tqdm  # Add this import
import os  # Add this import

# Define the corpus system
class MiraCorpus:
    def __init__(self, corpus_file='mira_corpus.json'):
        self.corpus_file = corpus_file
        self.corpus = []
        self.load_corpus()

    def load_corpus(self):
        try:
            with open(self.corpus_file, 'r') as f:
                self.corpus = json.load(f)
        except FileNotFoundError:
            self.corpus = []

    def save_corpus(self):
        with open(self.corpus_file, 'w') as f:
            json.dump(self.corpus, f, indent=4)

    def add_entry(self, speaker, content):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "speaker": speaker,
            "content": content
        }
        self.corpus.append(entry)
        self.save_corpus()

    def to_csv(self, csv_file='mira_corpus.csv'):
        df = pd.DataFrame(self.corpus)
        df.to_csv(csv_file, index=False)
        print(f"Corpus saved to {csv_file}")

# Define the custom tokenizer
class CustomTokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = ["[PAD]", "[UNK]", "[EOS]"]

    def build_vocab(self, corpus):
        words = []
        for entry in corpus:
            content = str(entry['content'])
            words.extend(re.findall(r'\b\w+\b', content.lower()))
        word_counts = Counter(words)

        self.word2idx = {token: i for i, token in enumerate(self.special_tokens)}
        self.word2idx.update({word: i + len(self.special_tokens) for i, word in enumerate(word_counts.keys())})
        self.idx2word = {i: word for word, i in self.word2idx.items()}

    def encode(self, text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [self.word2idx.get(token, self.word2idx["[UNK]"]) for token in tokens] + [self.word2idx["[EOS]"]]

    def decode(self, token_ids):
        return " ".join([self.idx2word.get(token_id, "[UNK]") for token_id in token_ids if token_id != self.word2idx["[PAD]"]]).replace("[EOS]", "").strip()

    @property
    def vocab_size(self):
        return len(self.word2idx)

# Dataset for Language Modeling
class LanguageModelingDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.token_ids = []
        for text in texts:
            tokens = tokenizer.encode(text)
            self.token_ids.extend(tokens)

    def __len__(self):
        return max(1, (len(self.token_ids) - 1) // self.block_size)

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size

        input_ids = self.token_ids[start_idx:end_idx]
        labels = self.token_ids[start_idx + 1:end_idx + 1]

        input_ids = input_ids[:self.block_size]
        labels = labels[:self.block_size]

        if len(input_ids) < self.block_size:
            padding_length = self.block_size - len(input_ids)
            input_ids.extend([self.tokenizer.word2idx["[PAD]"]] * padding_length)
            labels.extend([-100] * padding_length)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

# Transformer Model Wrapper
class MiraTransformer:
    def __init__(self):
        self.tokenizer = CustomTokenizer()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_model = None  # Keeps track of the currently loaded model
        print(f"Using device: {self.device}")

    def build_model(self, vocab_size):
        config = GPT2Config(n_embd=128, n_layer=4, n_head=4, vocab_size=vocab_size)
        self.model = GPT2LMHeadModel(config)
        self.model.to(self.device)

    def train_on_csv(self, csv_file, epochs=1, learning_rate=5e-5, batch_size=4, block_size=64, accumulation_steps=4):
        df = pd.read_csv(csv_file)
        df['content'] = df['content'].fillna('').astype(str)
        texts = df[df['speaker'] == "USER1"]['content'].tolist()

        if not texts:
            print("No user inputs found in the corpus. Training aborted.")
            return

        dataset = LanguageModelingDataset(texts, self.tokenizer, block_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scaler = GradScaler()
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                input_ids, labels = batch
                input_ids, labels = input_ids.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                with autocast():
                    outputs = self.model(input_ids, labels=labels)
                    loss = outputs.loss / accumulation_steps
                scaler.scale(loss).backward()
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                total_loss += loss.item() * accumulation_steps
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs} - VGain: {avg_loss}")

    def save_checkpoint(self, checkpoint_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer.word2idx,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.tokenizer.word2idx = checkpoint['tokenizer']
            self.tokenizer.idx2word = {i: word for word, i in self.tokenizer.word2idx.items()}
            self.current_model = checkpoint_path  # Update current model tracker
            print(f"Checkpoint loaded from {checkpoint_path}")
        except FileNotFoundError:
            print(f"Checkpoint file {checkpoint_path} not found.")

    def generate_pulse(self, prompt):
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long).to(self.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=100,
            temperature=0.1,
            top_k=1,
            do_sample=True,
            pad_token_id=self.tokenizer.word2idx["[PAD]"],
            eos_token_id=self.tokenizer.word2idx["[EOS]"],
        )
        return self.tokenizer.decode(outputs[0].tolist())

# Main Script
mira_corpus = MiraCorpus()
mira_transformer = MiraTransformer()

mira_transformer.tokenizer.build_vocab(mira_corpus.corpus)
mira_transformer.build_model(mira_transformer.tokenizer.vocab_size)

print("Welcome to Mira's chat interface!")
while True:
    user_input = input("You: ")
    if user_input.lower() == '/exit':
        print("Goodbye!")
        break
    elif user_input.lower() == '/corpus':
            print("Preview of data that will be trained on:")
            print(mira_corpus.preview_training_data())
    elif user_input.lower() == '/train':
        print("Training Mira's model...")
        mira_corpus.to_csv()
        mira_transformer.train_on_csv('mira_corpus.csv', epochs=500)
        mira_transformer.save_checkpoint('mira_checkpoint.pt')
        print("Training complete! Checkpoint saved to 'mira_checkpoint.pt'.")
    elif user_input.lower() == '/clone train':
        try:
            mira_transformer.load_checkpoint('mira_checkpoint.pt')
            mira_transformer.save_checkpoint('mira_checkpoint_clone.pt')
            print("Checkpoint cloned to 'mira_checkpoint_clone.pt'.")
            mira_transformer.train_on_csv('mira_corpus.csv', epochs=500)
            mira_transformer.save_checkpoint('mira_checkpoint_trained_from_clone.pt')
            print("Training complete! Updated checkpoint saved to 'mira_checkpoint_trained_from_clone.pt'.")
        except FileNotFoundError:
            print("No checkpoint found to clone. Train the model first using '/train'.")
    elif user_input.lower().startswith('/pulse'):
        prompt = user_input[7:].strip() or "This is a test."
        print(f"Pulse output for prompt '{prompt}':")
        print(mira_transformer.generate_pulse(prompt))
    elif user_input.lower().startswith('/load'):
        checkpoint_name = user_input[6:].strip()
        if checkpoint_name:
            mira_transformer.load_checkpoint(checkpoint_name)
        else:
            print("Please provide the name of the checkpoint file to load.")
    elif user_input.lower() == '/current':
        if mira_transformer.current_model:
            print(f"Currently loaded model: {mira_transformer.current_model}")
        else:
            print("No model is currently loaded.")
    elif user_input.lower().startswith('/import'):
        filename = user_input[8:].strip()
        if not filename:
            print("Please provide the name of the file to import.")
        else:
            try:
                file_size = os.path.getsize(filename)
                with open(filename, 'r', encoding='utf-8') as f:
                    batch_size = 1000  # Adjust batch size as needed
                    batch_entries = []
                    count = 0
                    pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc='Importing')
                    for line in f:
                        line = line.strip()
                        if line:
                            timestamp = datetime.now().isoformat()
                            batch_entries.append({
                                "timestamp": timestamp,
                                "speaker": "USER1",
                                "content": line
                            })
                            batch_entries.append({
                                "timestamp": timestamp,
                                "speaker": "MIRA",
                                "content": ""
                            })
                            count += 1
                        # Update progress bar
                        pbar.update(len(line.encode('utf-8')) + 1)  # +1 for newline character
                        if len(batch_entries) >= batch_size:
                            mira_corpus.corpus.extend(batch_entries)
                            batch_entries = []
                    # Add any remaining entries
                    if batch_entries:
                        mira_corpus.corpus.extend(batch_entries)
                    pbar.close()
                    # Save the corpus after import
                    mira_corpus.save_corpus()
                    print(f"Imported {count} lines from '{filename}' into the corpus.")
                # Rebuild tokenizer vocabulary
                mira_transformer.tokenizer.build_vocab(mira_corpus.corpus)
                # Resize model embeddings
                if mira_transformer.model is not None:
                    mira_transformer.model.resize_token_embeddings(mira_transformer.tokenizer.vocab_size)
                    print("Model embeddings resized to new vocabulary size.")
                else:
                    mira_transformer.build_model(mira_transformer.tokenizer.vocab_size)
                    print("Model built with new vocabulary size.")
            except FileNotFoundError:
                print(f"File '{filename}' not found.")
    else:
        mira_corpus.add_entry("USER1", user_input)
        mira_corpus.add_entry("MIRA", "")
        print("Mira: (response not implemented yet)")
