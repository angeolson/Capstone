# imports
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import random
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

SEED = 48
random.seed(48)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
glove = True

# ---------SET VARS--------------------
EPOCHS = 5
# MAX_LEN = 300
SEQUENCE_LEN = 4
BATCH_SIZE = 256
LR = 0.001
HIDDEN_SIZE = 128
NO_LAYERS = 4
DF_TRUNCATE_LB = 0  # lower bound to truncate data
DF_TRUNCATE_UB = 250  # upper bound to truncate data
Iterative_Train = False  # False if training model from scratch, True if fine-tuning
single_token_output = True  # True if only want to look at last word logits

# --------CLASS DEFINITIONS-------------
class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataframe,
            sequence_length,
            tokenizer,
            single_token_output,
            bert
    ):
        self.dataframe = dataframe
        self.single_token_output = single_token_output
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.bert = bert
        self.text = self.build_text(self.dataframe)

    def __len__(self):
        return len(self.text.split(' ')) - self.sequence_length # length of all inputs minus the sequence length, since that's our starting point

    def build_text(self, dataframe):
        dataframe['lyrics_added_tokens'] = dataframe['lyrics'].apply(lambda x:'[BOS] '+x+' [EOS]')
        input = dataframe['lyrics_added_tokens'].str.cat(sep=' ')
        input = input.replace('<newline>', '[SEP]')

        return input

    def build_sequences(self, text, sequence_length, max_len, single_token_output=True):
        '''
        adapted from https://gist.github.com/FernandoLpz/acaeb5fe714d084c0fe08481fdaa08b7#file-build_sequences-py
        :param text:
        :param word_to_index:
        :param sequence_length:
        :return:
        '''
        x_input_ids = list()
        y_input_ids = list()

        encoded_text = self.tokenizer.encode_plus(
            text=text,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=sequence_length,  # maximum length of a sentence = sequence
            pad_to_max_length=True,  # Add [PAD]s
            return_attention_mask=True  # Generate the attention mask
        )
        # encoded_text = self.tokenizer.encode(text)
        for i in range(max_len - (sequence_length + 1)):
            # try:
            # Get window of chars from text
            # Then, transform it into its idx representation
            input = encoded_text['input_ids'][i:i + sequence_length]

            # Get word target
            # Then, transform it into its idx representation
            if single_token_output is True:
                target = encoded_text['input_ids'][i + 1 + sequence_length]
            else:
                target = encoded_text['input_ids'][i + 1: i + 1 + sequence_length]  # longer than +1 token out

            # Save sequences and targets
            x_input_ids.append(input)
            y_input_ids.append(target)

        x_input_ids = np.array(x_input_ids)
        y_input_ids = np.array(y_input_ids)

        return x_input_ids, y_input_ids

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor from https://discuss.pytorch.org/t/is-there-something-like-keras-utils-to-categorical-in-pytorch/5960"""
        return np.eye(num_classes, dtype='uint8')[y['input_ids']]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        inputs = self.text.split(' ')[index:index + self.sequence_length]
        inputs = ' '.join(inputs)
        if self.single_token_output is True:
            targets = self.text.split(' ')[index + 1 + self.sequence_length]
            target_length = 1
        else:
            targets = self.text[index + 1: index + 1 + self.sequence_length]
            targets = ' '.join(targets)
            target_length = self.sequence_length

        x = self.tokenizer.encode_plus(
            text=inputs,  # the sentence to be encoded
            add_special_tokens=False,  # Do not add [CLS] and [SEP]
            max_length=self.sequence_length,  # maximum length of a sentence
            padding=False,  # Do not add padding
            return_attention_mask=False, # Do not generate the attention mask
            return_token_type_ids = False
        )

        y = self.tokenizer.encode_plus(
            text=targets,  # the sentence to be encoded
            add_special_tokens=False,  # Do not add [CLS] and [SEP]
            max_length=target_length,  # maximum length of a sentence
            padding=False,  # Do not add padding
            return_attention_mask=False, # Do not generate the attention mask
            return_token_type_ids=False
        )

        y = self.to_categorical(y=y, num_classes=self.bert.config.to_dict()['vocab_size'])  # equiv to n_vocab

        return torch.tensor(x['input_ids']), torch.tensor(y)


class Model(nn.Module):
    def __init__(self, bert, hidden_dim, no_layers, batch_size, single_token_output=True):
        super(Model, self).__init__()
        self.bert = bert
        self.hidden_dim = hidden_dim
        self.embedding_dim = bert.config.to_dict()['hidden_size']
        self.num_layers = no_layers
        self.n_vocab = bert.config.to_dict()['vocab_size']
        self.batch_size = batch_size
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_dim, self.n_vocab)
        self.single_token_output = single_token_output

    def forward(self, x, hidden):
        embed = self.bert(input_ids=x)[0]
        output, hidden = self.lstm(embed, hidden)
        out = self.fc(output)
        if self.single_token_output is True:
            out = out[:, -1, :]  # keeps only last logits, i.e. logits associated with the last word we want to predict
        return out, hidden

    def init_hidden(self):
        h0 = torch.zeros((self.num_layers, self.batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.num_layers, self.batch_size, self.hidden_dim)).to(device)
        return h0, c0


# --------------MODEL FUNCTIONS----------------
# def train(train_dataset, val_dataset, model, batch_size, max_epochs, seq_len):
def train(train_dataset, val_dataset, model, max_epochs, batch_size, lr):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    all_val_loss = []
    all_train_loss = []
    for epoch in range(max_epochs):
        train_losses = []
        val_losses = []
        train_batch_count = 0
        val_batch_count = 0
        model.train()
        state_h, state_c = model.init_hidden()
        for batch in train_dataloader:
            optimizer.zero_grad()
            X = batch[0].to(device)
            Y = batch[1].to(device)
            y_pred, (state_h, state_c) = model(X, (state_h, state_c))
            loss = criterion(y_pred, Y.float())
            state_h = state_h.detach()
            state_c = state_c.detach()
            loss.backward()
            train_losses.append(loss.item())
            print({'epoch': epoch, 'batch': train_batch_count, 'train loss': loss.item()})
            train_batch_count += 1
            optimizer.step()
        model.eval()
        val_state_h, val_state_c = model.init_hidden()
        for batch in val_dataloader:
            X = batch[0].to(device)
            Y = batch[1].to(device)
            y_pred, (val_state_h, val_state_c) = model(X, (val_state_h, val_state_c))
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
            torch.save(model.state_dict(), f"model_5_{DF_TRUNCATE_UB}_{single_token_output}.pt")
            print('model saved!')
    losses_df = pd.DataFrame()
    losses_df['val_loss'] = all_val_loss
    losses_df['train_loss'] = all_train_loss
    losses_df.to_csv(f'epoch_losses_m5_{DF_TRUNCATE_UB}_{single_token_output}.csv')


# ---------LOAD DATA--------------------
df = pd.read_csv('df_LSTM.csv', index_col=0)
df_copy = df.copy()
df_copy.reset_index(drop=True, inplace=True)

# truncate
df_copy = df.iloc[DF_TRUNCATE_LB:DF_TRUNCATE_UB]

# split data
train_, val_ = train_test_split(df_copy, train_size=0.8, random_state=SEED)

# export datasets
train_.to_csv(f'train_data_m5_{DF_TRUNCATE_UB}.csv')
val_.to_csv(f'val_data_m5_{DF_TRUNCATE_UB}.csv')

# -------------MODEL PREP----------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained("bert-base-uncased")

# freeze the pretrained layers
for param in bert.parameters():
    param.requires_grad = False

# add new tokens to tokenizer
new_tokens = ['<SONGBREAK>', '[BOS]', '[EOS]']
tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})  # add tokens for verses
bert.resize_token_embeddings(len(tokenizer))  # resize embeddings for added special tokens
unk_tok_emb = bert.embeddings.word_embeddings.weight.data[tokenizer.unk_token_id, :]  # get embedding for unknown token
for i in range(len(new_tokens)):  # initially apply that to all new tokens
    bert.embeddings.word_embeddings.weight.data[-(i + 1), :] = unk_tok_emb

train_dataset = Dataset(dataframe=train_, sequence_length=SEQUENCE_LEN, tokenizer=tokenizer,
                        single_token_output=single_token_output, bert=bert)
val_dataset = Dataset(dataframe=val_, sequence_length=SEQUENCE_LEN, tokenizer=tokenizer,
                      single_token_output=single_token_output, bert=bert)

model = Model(single_token_output=single_token_output, bert=bert, hidden_dim=HIDDEN_SIZE, no_layers=NO_LAYERS, batch_size=BATCH_SIZE).to(
    device)
if Iterative_Train is True:
    model.load_state_dict(torch.load(f'model_5_{DF_TRUNCATE_LB}_{single_token_output}.pt', map_location=device))

# ------------MODEL TRAIN----------------
train(train_dataset=train_dataset, val_dataset=val_dataset, model=model, max_epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)