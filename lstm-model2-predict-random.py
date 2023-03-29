# imports
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import os
import torch.nn.functional as F
import random

SEED = 48
random.seed(48)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define functions

# Generate a song
def text_generation(prompt_list):
    '''
    :param prompt_list: list
    :return: single song or list of songs; n returned = number of prompts given
    '''
    songs = []
    if len(prompt_list) > 1:
        for i in range(len(prompt_list)):
            prompt = prompt_list[i]
            song = generate(model.to('cpu'), tokenizer, prompt)
            songs.append(song)
        return songs
    else:
        song = generate(model.to('cpu'), tokenizer, prompt_list[0])
        return song


def generate(
        model,
        tokenizer,
        prompt,
        entry_length=250,  # maximum number of words
        top_p=0.8,
        temperature=1.,
):
    model.eval()
    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        entry_finished = False
        generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

        for i in range(entry_length):
            outputs = model(generated, labels=generated)
            loss, logits = outputs[:2]
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                ..., :-1
                                                ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value

            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

            if next_token in tokenizer.encode("<|endoftext|>"):
                entry_finished = True

            if entry_finished:
                generated_num = generated_num + 1

                output_list = list(generated.squeeze().numpy())
                output_text = tokenizer.decode(output_list)
                generated_list.append(output_text)
                break

        if not entry_finished:
            output_list = list(generated.squeeze().numpy())
            output_text = f"{tokenizer.decode(output_list)}<|endoftext|>"
            generated_list.append(output_text)

    return generated_list

#---------LOAD MODEL--------------------
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# new_tokens = ['<newline>', '<verse>', '<chorus>', '<prechorus>', '<bridge>', '<outro>', '<intro>', '<refrain>', '<hook>', '<postchorus>', '<other>']
new_tokens = ['<newline>', '<SONGBREAK>']
tokenizer.add_special_tokens({'additional_special_tokens': new_tokens}) # add tokens for verses
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer)) # resize embeddings for added special tokens
model.load_state_dict(torch.load('model_2.pt'))
model = model.to(device)


songs = text_generation(['here i am thinking about you', 'hey what do you know', 'cold dark days are ahead', "i'm not thinking about"])
print(songs)