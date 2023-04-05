# imports
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np
import random
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

SEED = 48
random.seed(48)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 250
DF_TRUNCATE_LB = 0  # lower bound to truncate data
DF_TRUNCATE_UB = 1000  # upper bound to truncate data
Iterative_Train = False  # False if training model from scratch, True if fine-tuning
single_token_output = False  # True if only want to look at last word logits
model_name = f'model_4_{DF_TRUNCATE_UB}_{single_token_output}.pt'

# --------CLASS DEFINITIONS-------------
class Model(nn.Module):
    def __init__(self, max_len, bert, hidden_dim, no_layers, single_token_output=True):
        super(Model, self).__init__()
        self.bert = bert
        self.hidden_dim = hidden_dim
        self.embedding_dim = bert.config.to_dict()['hidden_size']
        self.num_layers = no_layers
        self.n_vocab = bert.config.to_dict()['vocab_size']
        self.max_len = max_len
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(self.hidden_dim, 50)
        self.fc2 = nn.Linear(50, self.n_vocab)
        self.single_token_output = single_token_output

    def forward(self, x, hidden, x_attention):
        embed = self.bert(input_ids=x, attention_mask=x_attention)[0]
        output, hidden = self.lstm(embed, hidden)
        out = self.fc1(output)
        out = self.fc2(out)
        if self.single_token_output is True:
            out = out[:, -1, :]  # keeps only last logits, i.e. logits associated with the last word we want to predict
        # out = self.softmax(out)
        return out, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.num_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.num_layers, batch_size, self.hidden_dim)).to(device)
        return h0, c0

# -----------HELPER FUNCTIONS------------
def generate(
        model,
        tokenizer,
        prompt,
        single_token_output,
        entry_length=350 # maximum number of words

):
    model.eval()

    generated_lyrics = prompt.split(' ')

    with torch.no_grad():
        entry_finished = False
        state_h, state_c = model.init_hidden(1)
        # for i in range(entry_length):
        while len(generated_lyrics) < entry_length:
            generated = tokenizer.encode_plus(
                " ".join(generated_lyrics),
                add_special_tokens=False,
                return_token_type_ids=False,
                return_attention_mask=True,
                max_length=entry_length,  # maximum length of a song
                pad_to_max_length=True,
            )
            inputs = torch.tensor(generated['input_ids']).to(device)
            inputs = inputs.reshape(1, -1)
            attention_mask = torch.tensor(generated['attention_mask']).to(device)
            # mask = np.ones(len(generated_lyrics))
            # for i in range(len(generated_lyrics)):
            #     if generated_lyrics[i] == '[PAD]':
            #         mask[i] = 0
            #attention_mask = torch.tensor(mask).to(device)
            attention_mask = attention_mask.reshape(1, -1)
            y_pred, (state_h, state_c) = model(inputs, (state_h, state_c), attention_mask)
            if single_token_output is True:
                logits = y_pred[0]
            else:
                logits = y_pred[0][-1]
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_logits_prob = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_logits_prob, dim=-1)
            #sorted_indices_to_remove = cumulative_probs > 0.8
            sorted_indices_to_remove = sorted_logits_prob < 0.00001
            keep = sorted_indices[sorted_indices_to_remove]
            sorted_logits_prob_keep = sorted_logits_prob[:len(keep)]
            if len(sorted_logits_prob_keep) == 0:
                next_token = [0]  # padding token
            else:
                next_token_sorted = torch.multinomial(sorted_logits_prob_keep, num_samples=1)
                next_token = [keep[next_token_sorted].detach().cpu().numpy()[0]]

            # generated_lyrics = generated_lyrics + " " + tokenizer.decode(next_token)
            generated_lyrics.append(tokenizer.decode(next_token))

            if tokenizer.decode(next_token) == '[EOS]':
                entry_finished = True

            if entry_finished is True:
                break

    #generated_lyrics = generated_lyrics.replace('[PAD]', '')

    #return generated_lyrics
    return " ".join([item for item in generated_lyrics if item != '[PAD]'])

#---------LOAD MODEL--------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained("bert-base-uncased")
#freeze the pretrained layers
for param in bert.parameters():
    param.requires_grad = False

# add new tokens to tokenizer
new_tokens = ['<SONGBREAK>', '[BOS]', '[EOS]']
tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})  # add tokens for verses
bert.resize_token_embeddings(len(tokenizer))  # resize embeddings for added special tokens
unk_tok_emb = bert.embeddings.word_embeddings.weight.data[tokenizer.unk_token_id, :]  # get embedding for unknown token
for i in range(len(new_tokens)):  # initially apply that to all new tokens
    bert.embeddings.word_embeddings.weight.data[-(i + 1), :] = unk_tok_emb

model = Model(max_len=MAX_LEN, single_token_output=single_token_output, bert=bert, hidden_dim=128, no_layers=4).to(
    device)

model.load_state_dict(torch.load(model_name, map_location=device))

#------------MODEL RUN-----------------
song = generate(model=model, prompt="[BOS] <SONGBREAK> [SEP] i'm just walking in my shadows [SEP] where i'm walking to i don't know", entry_length=250, single_token_output=single_token_output, tokenizer=tokenizer)
print(song)