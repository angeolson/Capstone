import pandas as pd

model1 = pd.read_csv('epoch_losses_m1_hs256.csv')
model4 = pd.read_csv('epoch_losses_m4_all.csv')
model4_2 = pd.read_csv('epoch_losses_m4_hs128-2fc-vocab_trunc.csv')
model4_predictions = pd.read_csv('bert-lstm-gen.csv')

print('done')

