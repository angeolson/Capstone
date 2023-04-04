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
MAX_LEN = 350
glove = False

embedding_dim = 200  # set = 50 for the 50d file, eg.
filepath = f'Glove/glove.6B.{embedding_dim}d.txt'  # set filepath
single_token_output=True


# --------CLASS DEFINITIONS-------------

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataframe,
            sequence_length,
            tokenizer,
            max_len,
            single_token_output,
            bert
    ):
        self.dataframe = dataframe
        self.single_token_output = single_token_output
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.inputs, self.targets, self.inputs_attention = self.get_data(self.dataframe)
        self.batch_size = len(self.inputs) // len(self.dataframe)
        self.bert = bert

    def __len__(self):
        return len(self.inputs)

    def build_sequences(self, text, sequence_length, max_len, single_token_output=True):
        '''
        adapted from https://gist.github.com/FernandoLpz/acaeb5fe714d084c0fe08481fdaa08b7#file-build_sequences-py
        :param text:
        :param word_to_index:
        :param sequence_length:
        :return:
        '''
        x_input_ids = list()
        x_attention_mask = list()
        y_input_ids = list()
        y_attention_mask = list()
        text_length = len(text)
        if text_length < max_len:
            for i in range(max_len - text_length):
                text.append('[PAD]')
        # text = " ".join(text)

        # encoded_text = self.tokenizer.encode(text)
        for i in range(max_len - (sequence_length + 1)):
            # try:
            # Get window of chars from text
            # Then, transform it into its idx representation
            sequence = text[i:i + sequence_length]
            sequence = " ".join(sequence)
            encoded_sequence = self.tokenizer(
                sequence,
                add_special_tokens=False,
                max_length=sequence_length,
                return_token_type_ids=False,
                return_attention_mask=True,
                truncation=True,
                padding=True)
            # Get word target
            # Then, transform it into its idx representation

            if single_token_output is True:
                target = text[i + 1 + sequence_length]
                target = " ".join(target)
                encoded_target = self.tokenizer(
                    target,
                    add_special_tokens=False,
                    max_length=1,
                    return_token_type_ids=False,
                    return_attention_mask=False,
                    truncation=True,
                    padding=True)

            else:
                target = text[i + 1: i + 1 + sequence_length]  # longer than +1 token out
                target = " ".join(target)
                encoded_target = self.tokenizer(
                    target,
                    add_special_tokens=False,
                    max_length=sequence_length,
                    return_token_type_ids=False,
                    return_attention_mask=False,
                    truncation=True,
                    padding=True)

            # Save sequences and targets
            x_input_ids.append(encoded_sequence['input_ids'])
            x_attention_mask.append(encoded_sequence['attention_mask'])
            y_input_ids.append(encoded_target['input_ids'])

            # except:
            #     sequence = [word_to_index['PAD']]*sequence_length
            #     x.append(sequence)
            #     y.append(sequence)

        # x = np.array(x)
        # y = np.array(y)
        x_input_ids = np.array(x_input_ids)
        x_attention_mask = np.array(x_attention_mask)
        y_input_ids = np.array(y_input_ids)

        return x_input_ids, y_input_ids, x_attention_mask

    def get_data(self, dataframe):
        X_input_ids = list()
        X_attention_mask = list()
        Y_input_ids = list()

        for i in range(len(dataframe)):
            input = dataframe.iloc[i]['lyrics'] + ' [CLS]'
            input = input.replace('<newline>', '[SEP]')
            # input = dataframe.iloc[i]['lyrics']
            tokenized_input = input.split(" ")
            x, y, x_attention = self.build_sequences(text=tokenized_input,
                                                     sequence_length=self.sequence_length, max_len=self.max_len,
                                                     single_token_output=self.single_token_output)
            X_input_ids.append(x)
            Y_input_ids.append(y)
            X_attention_mask.append(x_attention)

        inputs = [sequence for song in X_input_ids for sequence in song]
        inputs_attention = [sequence for song in X_attention_mask for sequence in song]
        targets = [targ for song in Y_input_ids for targ in song]
        return np.array(inputs), np.array(targets), np.array(inputs_attention)

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor from https://discuss.pytorch.org/t/is-there-something-like-keras-utils-to-categorical-in-pytorch/5960"""
        return np.eye(num_classes, dtype='uint8')[y]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        x = self.inputs[index]
        x_attention = self.inputs_attention[index]
        y = self.targets[index]
        y = self.to_categorical(y=y, num_classes=self.bert.config.to_dict()['vocab_size'])  # equiv to n_vocab
        y = y[0]
        return torch.tensor(x), torch.tensor(y), torch.tensor(x_attention)


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
            dropout=0.2,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_dim, self.n_vocab)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2d = nn.Softmax2d()
        self.single_token_output = single_token_output

    def forward(self, x, hidden, x_attention):
        embed = self.bert(input_ids=x, attention_mask=x_attention)[0]
        output, hidden = self.lstm(embed, hidden)
        out = self.fc(output)
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
        entry_length=250 # maximum number of words

):
    model.eval()

    generated_lyrics = prompt.split(' ')

    with torch.no_grad():

        entry_finished = False
        state_h, state_c = model.init_hidden(1)
        # for i in range(entry_length):
        while len(generated_lyrics) < entry_length:
            generated = tokenizer(
                " ".join(generated_lyrics),
                add_special_tokens=False,
                return_token_type_ids=False,
                return_attention_mask=True,
                truncation=False,
                padding=False,
                pad_token=0)
            inputs = torch.tensor(generated['input_ids']).to(device)
            inputs = inputs.reshape(1, -1)
            attention_mask = torch.tensor(generated['attention_mask']).to(device)
            attention_mask = attention_mask.reshape(1, -1)
            y_pred, (state_h, state_c) = model(inputs, (state_h, state_c), attention_mask)
            if single_token_output is True:
                logits = y_pred[0]
            else:
                logits = y_pred[0][-1]
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_logits_prob = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_logits_prob, dim=-1)
            sorted_indices_to_keep = cumulative_probs < 0.8
            keep = sorted_indices[sorted_indices_to_keep]
            sorted_logits_prob_keep = sorted_logits_prob[:len(keep)]
            if len(sorted_logits_prob_keep) == 0:
                next_token = [0]  # padding token
            else:
                next_token_sorted = torch.multinomial(sorted_logits_prob_keep, num_samples=1)
                next_token = [keep[next_token_sorted].detach().cpu().numpy()[0]]

            # generated_lyrics = generated_lyrics + " " + tokenizer.decode(next_token)
            generated_lyrics.append(tokenizer.decode(next_token))

            if tokenizer.decode(next_token) == '[CLS]':
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
new_tokens = ['<SONGBREAK>']
tokenizer.add_special_tokens({'additional_special_tokens': new_tokens}) # add tokens for verses
bert.resize_token_embeddings(len(tokenizer)) # resize embeddings for added special tokens
unk_tok_emb = bert.embeddings.word_embeddings.weight.data[tokenizer.unk_token_id, :] # get embedding for unknown token
for i in range(len(new_tokens)): # initially apply that to all new tokens
        bert.embeddings.word_embeddings.weight.data[-(i+1), :] = unk_tok_emb

model = Model(max_len=MAX_LEN, single_token_output=single_token_output, bert=bert, hidden_dim=128, no_layers=4).to(device)

model.load_state_dict(torch.load('model_3_500_True.pt', map_location=device))

#------------MODEL RUN-----------------
song = generate(model=model, prompt="what are you doing tonight", entry_length=250, single_token_output=single_token_output, tokenizer=tokenizer)
print(song)
