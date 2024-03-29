# imports
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np
import random
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2Model, AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel

SEED = 48
random.seed(48)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
glove = True

# ---------SET VARS--------------------
EPOCHS = 5
MAX_LEN = 350
SEQUENCE_LEN = 4
DF_TRUNCATE_LB = 0  # lower bound to truncate data
DF_TRUNCATE_UB = 250  # upper bound to truncate data
Iterative_Train = False  # False if training model from scratch, True if fine-tuning
single_token_output = True  # True if only want to look at last word logits
# BATCH_SIZE = MAX_LEN - SEQUENCE_LEN

embedding_dim = 200  # set = 50 for the 50d file, eg.


# -----------HELPER FUNCTIONS------------
def cleanData(df):
    '''
    function cleans the original pandas dataframe imported from .csv and returns a clean frame
    :param df:
    :return: cleaned pandas dataframe
    '''
    for col in ['verses', 'verse_types', 'verses_transformed', 'EDA_verses']:
        df[col] = df[col].apply(lambda x: eval(x, {'__builtins__': None}, {}))
    return df


def load_words(dataframe):
    text = dataframe['lyrics'].str.cat(sep=' ')
    return text.split(' ')


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
                # target = " ".join(target)
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


# --------------MODEL FUNCTIONS----------------
# def train(train_dataset, val_dataset, model, batch_size, max_epochs, seq_len):
def train(train_dataset, val_dataset, model, max_epochs, seq_len):
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
    train_batch_size = train_dataset.batch_size
    val_batch_size = val_dataset.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=train_batch_size, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    all_val_loss = []
    all_train_loss = []
    for epoch in range(max_epochs):
        train_losses = []
        val_losses = []
        train_batch_count = 0
        val_batch_count = 0
        model.train()
        state_h, state_c = model.init_hidden(train_batch_size)
        optimizer.zero_grad()
        for batch in train_dataloader:
            optimizer.zero_grad()
            X = batch[0].to(device)
            Y = batch[1].to(device)
            X_Attention = batch[2].to(device)
            y_pred, (state_h, state_c) = model(X, (state_h, state_c), X_Attention)
            # loss = criterion(y_pred.transpose(1, 2), Y)
            loss = criterion(y_pred, Y.float())
            state_h = state_h.detach()
            state_c = state_c.detach()
            loss.backward()
            train_losses.append(loss.item())
            print({'epoch': epoch, 'batch': train_batch_count, 'train loss': loss.item()})
            # if (train_batch_count % 16) == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            train_batch_count += 1
            optimizer.step()
        model.eval()
        val_state_h, val_state_c = model.init_hidden(val_batch_size)
        for batch in val_dataloader:
            X = batch[0].to(device)
            Y = batch[1].to(device)
            X_Attention = batch[2].to(device)
            y_pred, (val_state_h, val_state_c) = model(X, (val_state_h, val_state_c), X_Attention)
            # loss = criterion(y_pred.transpose(1, 2), Y)
            val_loss = criterion(y_pred, Y.float())
            val_state_h = val_state_h.detach()
            val_state_c = val_state_c.detach()
            val_losses.append(val_loss.item())
            print({'epoch': epoch, 'batch': val_batch_count, 'val loss': val_loss.item()})
            val_batch_count += 1

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        all_val_loss.append(epoch_val_loss)
        all_train_loss.append(epoch_train_loss)
        print(f'Epoch {epoch}')
        print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
        best_loss = min(all_val_loss)
        if epoch_val_loss <= best_loss:
            torch.save(model.state_dict(), f"model_3_{DF_TRUNCATE_UB}_{single_token_output}_seq4.pt")
            print('model saved!')
    losses_df = pd.DataFrame()
    losses_df['val_loss'] = all_val_loss
    losses_df['train_loss'] = all_train_loss
    losses_df.to_csv(f'epoch_losses_m3_{DF_TRUNCATE_UB}_{single_token_output}_seq4.csv')


# ---------LOAD DATA--------------------
df = pd.read_csv('df_LSTM.csv', index_col=0)
df_copy = df.copy()
df_copy.reset_index(drop=True, inplace=True)

# truncate
df_copy = df.iloc[DF_TRUNCATE_LB:DF_TRUNCATE_UB]

# split data
train_, val_ = train_test_split(df_copy, train_size=0.8, random_state=SEED)

# export datasets
train_.to_csv(f'train_data_m3_{DF_TRUNCATE_UB}_seq4.csv')
val_.to_csv(f'val_data_m3_{DF_TRUNCATE_UB}_seq4.csv')

# -------------MODEL PREP----------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained("bert-base-uncased")
# freeze the pretrained layers
for param in bert.parameters():
    param.requires_grad = False

# add new tokens to tokenizer
new_tokens = ['<SONGBREAK>']
tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})  # add tokens for verses
bert.resize_token_embeddings(len(tokenizer))  # resize embeddings for added special tokens
unk_tok_emb = bert.embeddings.word_embeddings.weight.data[tokenizer.unk_token_id, :]  # get embedding for unknown token
for i in range(len(new_tokens)):  # initially apply that to all new tokens
    bert.embeddings.word_embeddings.weight.data[-(i + 1), :] = unk_tok_emb

train_dataset = Dataset(dataframe=train_, sequence_length=SEQUENCE_LEN, tokenizer=tokenizer, max_len=MAX_LEN,
                        single_token_output=single_token_output, bert=bert)
val_dataset = Dataset(dataframe=val_, sequence_length=SEQUENCE_LEN, tokenizer=tokenizer, max_len=MAX_LEN,
                      single_token_output=single_token_output, bert=bert)

model = Model(max_len=MAX_LEN, single_token_output=single_token_output, bert=bert, hidden_dim=128, no_layers=4).to(
    device)
if Iterative_Train is True:
    model.load_state_dict(torch.load(f'model_3_{DF_TRUNCATE_LB}_{single_token_output}_seq4.pt', map_location=device))

# ------------MODEL TRAIN----------------
# train(train_dataset=train_dataset, val_dataset=val_dataset, model=model, batch_size=BATCH_SIZE, max_epochs=EPOCHS, seq_len=SEQUENCE_LEN)
train(train_dataset=train_dataset, val_dataset=val_dataset, model=model, max_epochs=EPOCHS, seq_len=SEQUENCE_LEN)
