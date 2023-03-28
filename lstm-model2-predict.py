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

SEED = 48
random.seed(48)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# define functions

# Function to generate multiple sentences. Test data should be a dataframe
def text_generation(test_data):
    generated_lyrics = []
    for i in range(len(test_data)):
        x = generate(model.to('cpu'), tokenizer, test_data['lyrics'][i], entry_count=1)
        generated_lyrics.append(x)
    return generated_lyrics


def generate(
        model,
        tokenizer,
        prompt,
        entry_count=10,
        entry_length=30,  # maximum number of words
        top_p=0.8,
        temperature=1.,
):
    model.eval()
    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in trange(entry_count):

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
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.load_state_dict(torch.load('model_2.pt', map_location=device))
model = model.to(device)

#---------LOAD TEST DATA--------------------
test_ = pd.read_csv('test_data_m2.csv')
test_ = test_[0:50]
test_.reset_index(drop=True, inplace=True)

#For the test set only, keep last 20 words in a new column, then remove them from original column
test_['True_end_lyrics'] = test_['lyrics'].str.split().str[-20:].apply(' '.join)

# generate on test set
#Loop to keep only generated text and add it as a new column in the dataframe
my_generations=[]

# Run the functions to generate the lyrics
generated_lyrics = text_generation(test_)

for i in range(len(generated_lyrics)):
  a = test_['lyrics'][i].split()[-30:] #Get the matching string we want (30 words)
  b = ' '.join(a)
  c = ' '.join(generated_lyrics[i]) #Get all that comes after the matching string
  my_generations.append(c.split(b)[-1])

test_['Generated_lyrics'] = my_generations


#Finish the sentences when there is a point, remove after that
final=[]

for i in range(len(test_)):
  to_remove = test_['Generated_lyrics'][i].split('.')[-1]
  final.append(test_['Generated_lyrics'][i].replace(to_remove,''))

test_['Generated_lyrics'] = final