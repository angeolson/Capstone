# Capstone: Modeling 

This folder contains project files associated with my George Washington Univeristy MS in Data Science Captstone, specifically in model training and lyric generation. 

If you are only interested in generating songs, assuming you have downloaded the model (link here), run either mass_generate.py to generate many songs, or generate.py to generate a single song. 

If you would like to train the model yourself, you will need to first run train.py. dataset.py and model.py contain class definitions; both are needed to run training, only model.py is needed to generate. 

Below, I describe the arguments needed to run the files in this folder at the command line. For more detail (data type, defualt values) please see the files themselves.

**train.py arguments**
* "-dt_pth": path to the data files, required
* "-ex_pth": path for exported files, required
* "-e": number of epochs to run training for, required
* "-mlen": max length to truncate songs, required
* "-slen": sequence length for inputs and outputs, required
* "-lr": learning rate", required
* "-trunc": if you would like to truncate the dataset, required
* "-lb": truncation lower bound, NOT required
* "-ub": "truncation upper bound, NOT required 
* "-iter": True if fine-tuning or training off previous checkpoint, False if training from scratch, required
* "-sing": True if only want to look at last word logits while calculating loss, required
* "-load": If fine-tuning or training off previous checkpoint, load model name, NOT required
* "-save": name of model to save, required
* "-file": name of file to save epoch loss data to, required
* "-v": True if want to see loss for each batch for each epoch, required

**generate.py arguments**
* "-m_pth": path to the model file, required
* "-m": name of model to use in generation, required
* "-sing": True if only want to look at last word logits, required
* "-mlen": max length to truncate songs, required
* "-t": temperature: recommend selecting between 0.9 and 1.1, required
* -p": single word prompter, required

**mass_generate.py arguments**
* "-m_pth": path to the model file,  required
* "-m": name of model to use in generation, required
* "-ex": name of file to export songs to, required
* "-ex_pth": path to export songs to, required
* "-sing": True if only want to look at last word logits, required
* "-mlen": low bound max length to truncate songs, required
* "-hmlen": high bound max length to truncate songs, required
* "-lt": temperature low: recommend selecting between 0.9 and 1.1, required
* "-ht": temperature high: recommend selecting between 0.9 and 1.1, required
* "-p": list of prompt words, required
* "-r": how many songs to generate, required




