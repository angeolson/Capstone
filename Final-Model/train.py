# imports
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import random
from transformers import BertTokenizer, BertModel, BertConfig
from dataset import Dataset
from model import Model
import os
import argparse

SEED = 48
random.seed(48)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------SET VARS--------------------
parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("-dt_pth", "--data_path", default = 'None', type=str, help = "path to the data files", required=True)
parser.add_argument("-ex_pth", "--export_path", default = 'None', type=str, help = "path for exported files", required=True)
parser.add_argument("-e", "--epochs", default = 50, type=int, help = "number of epochs to run training for", required=True)
parser.add_argument("-mlen", "--max_len", default = 250, type=int, help = "max length to truncate songs", required=True)
parser.add_argument("-slen", "--seq_len", default = 250, type=int, help = "sequence length for inputs and outputs", required=True)
parser.add_argument("-lr", "--learning_rate", default = 0.001, type=int, help = "learning rate", required=True)
parser.add_argument("-trunc", "--truncate", default = False, type=bool, help = "if you would like to truncate the dataset", required=True)
parser.add_argument("-lb", "--lower_bound", default = 0, type=int, help = "truncation lower bound", required=False)
parser.add_argument("-ub", "--upper_bound", default = 0, type=int, help = "truncation upper bound", required=False)
parser.add_argument("-iter", "--iterative_train", default = False, type=bool, help = "True if fine-tuning or training off stopping point, False if training from scratch", required=True)
parser.add_argument("-sing", "--single_token", default = False, type=bool, help = "True if only want to look at last word logits", required=True)
parser.add_argument("-load", "--load_model", default = 'None', type=str, help = "If fine-tuning or training off stopping point, load previous model", required=False)
parser.add_argument("-save", "--save_model", default = 'model-4-hs128-2fc-vocab_trunc.pt', type=str, help = "name of model to save", required=True)
parser.add_argument("-file", "--loss_file", default = 'epoch_losses_m4_hs128-2fc-vocab_trunc.csv', type=str, help = "name of file to save epoch loss data to", required=True)
parser.add_argument("-v", "--verbose", default = False, type=bool, help = "True if want to see loss for each batch for each epoch", required=True)
args = vars(parser.parse_args())

EPOCHS = args['epochs']
MAX_LEN = args['max_len']
SEQUENCE_LEN = args['seq_len']
LR =  args['learning_rate']
TRUNCATE = args['truncate']
DF_TRUNCATE_LB = args['lower_bound']
DF_TRUNCATE_UB = args['upper_bound']
Iterative_Train = args['iterative_train']
single_token_output = args['single_token']
load_model = args['load_model']
save_model = args['save_model']
filepath_for_losses = args['file']
DATA_PATH = args['data_path']
EXPORT_PATH = args['export_path']
VERBOSE = args['verbose']

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

# --------------TRAIN FUNCTION----------------
def train(train_dataset, val_dataset, model, max_epochs, lr):
    train_batch_size = train_dataset.batch_size
    val_batch_size = val_dataset.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=train_batch_size, drop_last=True)
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
        state_h, state_c = model.init_hidden(train_batch_size)
        for batch in train_dataloader:
            optimizer.zero_grad()
            X = batch[0].to(device)
            Y = batch[1].to(device)
            X_Attention = batch[2].to(device)
            y_pred, (state_h, state_c) = model(X, (state_h, state_c), X_Attention)
            outputs = y_pred.view(4* train_batch_size, 30525)
            targets = Y.view(-1)
            if len(targets) == 4* train_batch_size:
                loss = criterion(outputs, targets)
                state_h = state_h.detach()
                state_c = state_c.detach()
                loss.backward()
                train_losses.append(loss.item())
                if VERBOSE is True:
                    print({'epoch': epoch, 'batch': train_batch_count, 'train loss': loss.item()})
                train_batch_count += 1
            optimizer.step()
        model.eval()
        val_state_h, val_state_c = model.init_hidden(val_batch_size)
        for batch in val_dataloader:
            X = batch[0].to(device)
            Y = batch[1].to(device)
            X_Attention = batch[2].to(device)
            y_pred, (val_state_h, val_state_c) = model(X, (val_state_h, val_state_c), X_Attention)
            outputs = y_pred.view(4 * train_batch_size, 30525)
            targets = Y.view(-1)
            if len(targets) == 4 * train_batch_size:
                val_loss = criterion(outputs, targets)
                val_state_h = val_state_h.detach()
                val_state_c = val_state_c.detach()
                val_losses.append(val_loss.item())
                if VERBOSE is True:
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
            torch.save(model.state_dict(), save_model)
            print('model saved!')
        losses_df = pd.DataFrame()
        losses_df['val_loss'] = all_val_loss
        losses_df['train_loss'] = all_train_loss
        losses_df.to_csv(filepath_for_losses)

if __name__ == '__main__':
    # ---------LOAD DATA--------------------
    os.chdir(DATA_PATH)
    df_train = pd.read_csv('training.csv', index_col=0)
    df_train.reset_index(drop=True, inplace=True)

    df_val = pd.read_csv('validation.csv', index_col=0)
    df_val.reset_index(drop=True, inplace=True)

    # truncate; 0.2*250 = 50
    if TRUNCATE is True:
        train_ = df_train.iloc[DF_TRUNCATE_LB:DF_TRUNCATE_UB]
        val_ = df_val.iloc[DF_TRUNCATE_LB:int(DF_TRUNCATE_UB*0.2)]
    else:
        train_ = df_train.copy()
        val_ = df_val.copy()

    # -------------MODEL PREP----------------
    configuration = BertConfig(vocab_size=25000)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel(config=configuration).from_pretrained("bert-base-uncased")

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

    train_dataset = Dataset(dataframe=train_, sequence_length=SEQUENCE_LEN, tokenizer=tokenizer, max_len=MAX_LEN,
                            single_token_output=single_token_output, bert=bert)
    val_dataset = Dataset(dataframe=val_, sequence_length=SEQUENCE_LEN, tokenizer=tokenizer, max_len=MAX_LEN,
                          single_token_output=single_token_output, bert=bert)

    model = Model(max_len=MAX_LEN, single_token_output=single_token_output, bert=bert, hidden_dim=128, no_layers=4).to(
        device)
    if Iterative_Train is True:
        model.load_state_dict(torch.load(load_model, map_location=device))

    # ------------MODEL TRAIN----------------
    os.chdir(EXPORT_PATH)
    # train(train_dataset=train_dataset, val_dataset=val_dataset, model=model, batch_size=BATCH_SIZE, max_epochs=EPOCHS, seq_len=SEQUENCE_LEN)
    train(train_dataset=train_dataset, val_dataset=val_dataset, model=model, max_epochs=EPOCHS, lr=LR)