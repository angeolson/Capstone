import pandas as pd

model1 = pd.read_csv('epoch_losses_m1_all.csv')
model4 = pd.read_csv('epoch_losses_m4_all.csv')
model4_next10 = pd.read_csv('epoch_losses_m4_all_next10.csv')
model4_predictions = pd.read_csv('bert-lstm-gen.csv')

print('done')