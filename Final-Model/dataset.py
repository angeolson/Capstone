import torch
from torch.utils.data import DataLoader
import numpy as np

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

        encoded_text = self.tokenizer.encode_plus(
            text=text,  # the sentence to be encoded
            add_special_tokens=False,  # Don't add [CLS] and [SEP]
            max_length=max_len,  # maximum length of a song
            pad_to_max_length=True,  # Add [PAD]s
            return_attention_mask=True  # Generate the attention mask
        )
        # encoded_text = self.tokenizer.encode(text)
        for i in range(max_len - (sequence_length + 1)):
            # try:
            # Get window of chars from text
            # Then, transform it into its idx representation
            input = encoded_text['input_ids'][i:i + sequence_length]
            input_attention = encoded_text['attention_mask'][i:i + sequence_length]

            # Get word target
            # Then, transform it into its idx representation
            if single_token_output is True:
                target = encoded_text['input_ids'][i + 1 + sequence_length]
            else:
                target = encoded_text['input_ids'][i + 1: i + 1 + sequence_length]  # longer than +1 token out

            # Save sequences and targets
            x_input_ids.append(input)
            x_attention_mask.append(input_attention)
            y_input_ids.append(target)

        x_input_ids = np.array(x_input_ids)
        x_attention_mask = np.array(x_attention_mask)
        y_input_ids = np.array(y_input_ids)

        return x_input_ids, y_input_ids, x_attention_mask

    def get_data(self, dataframe):
        X_input_ids = list()
        X_attention_mask = list()
        Y_input_ids = list()

        for i in range(len(dataframe)):
            input = '[BOS] ' + dataframe.iloc[i]['lyrics'] + ' [EOS]'
            input = input.replace('<newline>', '[SEP]')

            x, y, x_attention = self.build_sequences(text=input,
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
        #y = self.to_categorical(y=y, num_classes=self.bert.config.to_dict()['vocab_size'])  # equiv to n_vocab
        #y = y[0]
        return torch.tensor(x), torch.tensor(y), torch.tensor(x_attention)