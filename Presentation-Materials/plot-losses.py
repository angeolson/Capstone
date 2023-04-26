# imports
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# set vars
sns.set_theme()
# argparse vars
parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("-dt_pth", "--input_path", default = 'None', type=str, help = "path to the data files")
parser.add_argument("-dt_nme", "--input_name", default = 'None', type=str, help = "name of data file")
parser.add_argument("-ex_pth", "--export_path", default = 'None', type=str, help = "path for final plot")
parser.add_argument("-ex_nme", "--export_name", default = 'None.csv', type=str, help = "name for final plot")
args = vars(parser.parse_args())
IMGPATH = args['export_path']
DATAPATH = args['input_path']
FILE = args['input_name']
fig_filename = args['export_name']

if __name__ == "__main__":
    os.chdir(DATAPATH)
    df = pd.read_csv(FILE)
    df = pd.melt(df, id_vars='Unnamed: 0', value_vars=['val_loss', 'train_loss'])

    os.chdir(IMGPATH)
    fig = plt.figure()
    ax = sns.lineplot(
        df, x="Unnamed: 0", y="value", hue="variable"
    )
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.set(title='Training and Validation Loss')
    fig.savefig(fig_filename, bbox_inches='tight', dpi=200)
    plt.show()