import os 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

INPUT_PATH_ = os.getcwd()
full_data = pd.read_csv('real_songs_sample.csv')
full_data = full_data[['artist', 'title', 'ID']]
full_data['Generated'] = full_data['artist'] == 'GEN'


def getDataframe(starting_df):
    '''
    :param type: either train or test split
    :return: pandas dataframe
    '''
    df = pd.DataFrame() # init df
    participants = 0
    for i in range(1,7):
        filepath = INPUT_PATH_ + f'/group_{i}'
        files = os.listdir(filepath)
        files = [item for item in files if item != '.DS_Store']
        print(f'group {i}'), print(files)
        participants += len(files)
        for file in files:
            file_df = pd.read_csv(filepath+'/'+file, encoding = 'unicode_escape')
            df = pd.concat([df,file_df])
    df.reset_index(drop=True)
    df = df[['ID', 'Score']]
    averages = df.groupby('ID')['Score'].mean()
    counts = df.groupby('ID')['Score'].count()
    df_merged = pd.merge(averages, counts, on='ID').rename(columns={'Score_x': 'Mean', 'Score_y': 'Count'})
    print(f'Participants: {participants}')
    return pd.merge(starting_df, df_merged, on='ID')

df = getDataframe(full_data)

fig = plt.figure()
ax = sns.histplot(data=df, x='Mean', hue='Generated')
ax.set_ylabel('')
ax.set_xlabel('Confidence Score')
ax.set_title('Survey Results')
fig.savefig('Song_Gens.png', bbox_inches='tight', dpi=200)
plt.show()

mean_gen = df[df['Generated'] == True]['Mean'].mean()
count_gen = df[df['Generated'] == True]['Count'].sum()
count_gen_songs = len(df[df['Generated'] == True])

mean_real = df[df['Generated'] == False]['Mean'].mean()
count_real = df[df['Generated'] == False]['Count'].sum()
count_real_songs = len(df[df['Generated'] == False])

print(f'Average Score for Gen Song: {mean_gen}')
print(f'Participant Views for Gen Song: {count_gen}')
print(f'Number of Gen Songs: {count_gen_songs}')

print(f'Average Score for Real Song: {mean_real}')
print(f'Participant Views for Real Song: {count_real}')
print(f'Number of Real Songs: {count_real_songs}')