# imports
import torch
import random
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import numpy as np
from model import Model
import os
import argparse
import pandas as pd

# set vars
SEED = 48
random.seed(48)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("-m_pth", "--model_path", default = 'None', type=str, help = "path to the model file", required=True)
parser.add_argument("-m", "--model", default = 'model-4-hs128-2fc-vocab_trunc.pt', type=str, help = "name of model to use in generation", required=True)
parser.add_argument("-ex", "--export_file", default = 'bert-lstm-gen_trunc_model.csv', type=str, help = "name of file to export songs to", required=True)
parser.add_argument("-ex_pth", "--export_path", default = 'None', type=str, help = "path to export songs to", required=True)
parser.add_argument("-sing", "--single_token", default = False, type=bool, help = "True if only want to look at last word logits", required=True)
parser.add_argument("-mlen", "--low_max_len", default = 250, type=int, help = "low bound max length to truncate songs", required=True)
parser.add_argument("-hmlen", "--high_max_len", default = 350, type=int, help = "high bound max length to truncate songs", required=True)
parser.add_argument("-lt", "--low_temp", default = 0.9, type=float, help = "temperature low: recommend selecting between 0.9 and 1.1", required=True)
parser.add_argument("-ht", "--high_temp", default = 1, type=float, help = "temperature high: recommend selecting between 0.9 and 1.1", required=True)
parser.add_argument("-p", "--prompt_list", default = ['darkness', 'love', 'i', 'you', 'help', 'what', 'have', 'in', 'once', 'who', 'tomorrow', 'today', 'let'],  nargs='+', help = "list of prompt words", required=True)
parser.add_argument("-r", "--range", default = 100, type=int, help = "how many songs to generate", required=True)
args = vars(parser.parse_args())

MAX_LEN = args['low_max_len']
HIGH_MAX_LEN = args['high_max_len']
MODEL_PATH = args['model_path']
single_token_output = args['single_token']
model_name = args['model']
low_temperature_ = args['low_temp']
high_temperature = args['high_temp']
prompt_list = args['prompt_list']
export_file = args['export_file']
EXPORT_PATH = args['export_path']
RANGE = args['range']

# -----------GENERATE FUNCTION------------
def generate(
        model,
        tokenizer,
        prompt,
        single_token_output,
        entry_length=350,
        temperature=1.0

):
    '''
    temperature calc adapted from https://github.com/klaudia-nazarko/nlg-text-generation/blob/main/LSTM_class.py
    :param model:
    :param tokenizer:
    :param prompt:
    :param single_token_output:
    :param entry_length:
    :param temperature:
    :return:
    '''

    model.eval()

    generated_lyrics = prompt.split(' ')

    with torch.no_grad():
        entry_finished = False
        state_h, state_c = model.init_hidden(1)
        while len(generated_lyrics) < entry_length:
            generated = tokenizer.encode_plus(
                " ".join(generated_lyrics),
                add_special_tokens=False,  # Don't add [CLS] and [SEP]
                return_attention_mask=False  # Generate the attention mask
            )
            inputs = torch.tensor(generated['input_ids'][-4:]).to(device)
            input_list = list(inputs.detach().cpu().numpy())
            mask = [int((tokenizer.decode(el)) != '[PAD]') for el in input_list]
            inputs = inputs.reshape(1, -1)
            attention_mask = torch.tensor(mask).to(device)
            attention_mask = attention_mask.reshape(1, -1)
            y_pred, (state_h, state_c) = model(inputs, (state_h, state_c), attention_mask)
            if single_token_output is True:
                logits = y_pred[0]
            else:
                logits = y_pred[0][-1]
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_logits_prob = F.softmax(sorted_logits, dim=-1)
            prob_with_temperature = np.exp(np.where(sorted_logits_prob.detach().cpu().numpy() == 0, 0, np.log(sorted_logits_prob.detach().cpu().numpy() + 1e-10)) / temperature)
            prob_with_temperature /= np.sum(prob_with_temperature)
            # cumulative_probs = torch.cumsum(sorted_logits_prob, dim=-1)
            # sorted_indices_to_remove = cumulative_probs > 0.8
            # keep = sorted_indices[sorted_indices_to_remove]
            # sorted_logits_prob_keep = sorted_logits_prob[:len(keep)]
            # if len(sorted_logits_prob_keep) == 0:
            #     next_token = [0]  # padding token
            # else:
            #     next_token_sorted = torch.multinomial(sorted_logits_prob_keep, num_samples=1)
            #     next_token = [keep[next_token_sorted].detach().cpu().numpy()[0]]
            next_token_sorted = torch.multinomial(torch.tensor(prob_with_temperature), num_samples=1)
            next_token = [sorted_indices[next_token_sorted].detach().cpu().numpy()[0]]

            # generated_lyrics = generated_lyrics + " " + tokenizer.decode(next_token)
            generated_lyrics.append(tokenizer.decode(next_token))

            if tokenizer.decode(next_token) == '[EOS]':
                entry_finished = True

            # if tokenizer.decode(next_token) == '[PAD]':
            #     mask.append(0)
            # else:
            #     mask.append(1)

            if entry_finished is True:
                break

    #generated_lyrics = generated_lyrics.replace('[PAD]', '')

    #return generated_lyrics
    return " ".join([item for item in generated_lyrics if item != '[PAD]'])

#---------LOAD MODEL--------------------
if __name__ == '__main__':
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

    os.chdir(MODEL_PATH)
    model.load_state_dict(torch.load(model_name, map_location=device))

    #------------MODEL RUN-----------------
    song_list = [ ]
    for i in range(RANGE):
        prompt = random.choice(prompt_list)
        temp = np.random.uniform(low=low_temperature_, high=high_temperature, size=None)
        entry_length = np.random.randint(MAX_LEN, high=HIGH_MAX_LEN, size=None, dtype=int)
        song = generate(model=model, prompt=f"[BOS] <SONGBREAK> [SEP] {prompt}", entry_length=entry_length, single_token_output=single_token_output, tokenizer=tokenizer, temperature = temp)
        song_list.append(song)

    os.chdir(EXPORT_PATH)
    generated_songs = pd.DataFrame()
    generated_songs['song'] = song_list
    generated_songs['gen_type'] = 'lstm-bert'
    generated_songs.to_csv(export_file  )