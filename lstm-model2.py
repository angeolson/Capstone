# Adapted from https://towardsdatascience.com/how-to-fine-tune-gpt-2-for-text-generation-ae2ea53bc272, github https://github.com/francoisstamant/lyrics-generation-with-GPT2

# imports
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

SEED = 48
random.seed(48)

# define helper functions
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None

# define classes
class SongLyrics(Dataset):
    def __init__(self, lyric_list,  tokenizer, truncate=False, max_length=1024):

        self.tokenizer = tokenizer
        self.lyrics = []

        for row in lyric_list:
            self.lyrics.append(torch.tensor(
                self.tokenizer.encode(f"{row[:max_length]}<|endoftext|>")
            ))
        if truncate:
            self.lyrics = self.lyrics[:20000]
        self.lyrics_count = len(self.lyrics)

    def __len__(self):
        return self.lyrics_count

    def __getitem__(self, item):
        return self.lyrics[item]

def train(
    dataset, model,
    batch_size=16, epochs=5, lr=2e-5,
    warmup_steps=200):
    acc_steps = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    train_loss = []
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):
        print(f"Training epoch {epoch}")
        epoch_losses = []
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()
            epoch_losses.append(loss.item())
            #print(loss.item())

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        epoch_train_loss = np.mean(epoch_losses)
        print(f'epoch train loss: {epoch_train_loss}')
        train_loss.append(epoch_train_loss)
        best_train_loss = max(train_loss)
        if epoch_train_loss <= best_train_loss:
            torch.save(model.state_dict(), "model_2.pt")
            print('model saved!')
    return model

#Get the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
new_tokens = ['<newline>', '<verse>', '<chorus>', '<prechorus>', '<bridge>', '<outro>', '<intro>', '<refrain>', '<hook>', '<postchorus>', '<other>']
tokenizer.add_special_tokens({'additional_special_tokens': new_tokens}) # add tokens for verses
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer)) # resize embeddings for added special tokens
unk_tok_emb = model.transformer.wte.weight.data[tokenizer.unk_token_id, :] # get embedding for unknown token
for i in range(len(new_tokens)): # initially apply that to all new tokens
        model.transformer.wte.weight.data[-(i+1), :] = unk_tok_emb

# load data
df = pd.read_csv('df_LSTM.csv', index_col=0)
df_copy = df.copy()
df_copy.reset_index(drop=True, inplace=True)

# split data
train_, test_ = train_test_split(df_copy, train_size=0.8, random_state=SEED)
train_, val_ = train_test_split(train_, train_size=0.8, random_state=SEED)

# export datasets
train_.to_csv('train_data_m2.csv')
val_.to_csv('val_data_m2.csv')
test_.to_csv('test_data_m2.csv')

train_.reset_index(drop=True, inplace=True)
val_.reset_index(drop=True, inplace=True)

# create datasets
train_dataset = SongLyrics(train_['lyrics'], truncate=True, tokenizer=tokenizer)
val_dataset = SongLyrics(val_['lyrics'], truncate=True, tokenizer=tokenizer)

# train model
model = train(train_dataset, model)

