import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_normal = pd.read_fwf('./logs/resnet50.log', skiprows=1, skipfooter=1, widths=[24, 5, 9, 14, 11, 13, 10])
df_adv = pd.read_fwf('./logs/resnet50_pgd.log', skiprows=1, skipfooter=1, widths=[24, 5, 9, 14, 11, 13, 10])

print(df_normal.head())
print(df_normal.columns)

normal = df_normal[['Train Acc', 'Test Acc']]
normal = normal.rename(columns={'Train Acc': 'Normal Training Accuracy', 'Test Acc': 'Normal Testing Accuracy'})

adv = df_adv[['Train Acc', 'Test Acc']]
adv = adv.rename(columns={'Train Acc': 'Adv. Training Accuracy', 'Test Acc': 'Adv. Testing Accuracy'})

plot = pd.concat([normal, adv], axis=1)

plot = plot.plot.line()
fig = plot.get_figure()
fig.savefig("plots.png")

# # first 40 epochs
# x=range(1,21)
# y_train=[21.8962, 28.0373, 32.4898, 37.1199, 41.3762, 45.6433, 49.2569, 52.3545, 55.6977, 58.5651, 
#          61.0292, 63.5336, 65.2345, 67.4808, 68.9130, 70.2308, 71.6290, 72.6981, 73.6961, 74.6246, 
# ]
# y_val  =[30.9914, 39.1810, 42.4569, 46.7672, 52.2845, 60.3017, 60.9483, 65.4741, 67.8448, 69.0517, 
#          73.1897, 72.8448, 75.6466, 78.1034, 79.7414, 80.2586, 80.6897, 79.4397, 82.2414, 81.8534, 
# ]

# y_adv_train=[14.0326, 16.8351, 17.6060, 17.9938, 18.7415, 19.2451, 19.7272, 20.3868, 20.7190, 21.0048, 
#              21.6769, 22.2964, 22.7985, 23.4165, 24.2105, 24.5906, 25.3677, 26.4152, 26.9003, 27.1583, 
# ]
# y_adv_val=[23.4483, 25.5172, 26.0345, 30.8621, 28.4914, 32.1983, 30.9483, 35.4741, 37.8879, 37.0259, 
#            40.6466, 39.5690, 41.8966, 45.0862, 46.1638, 48.7069, 49.1379, 53.7500, 53.0603, 55.6466
# ]

# plt.plot(x, y_train, label='training acc', color='tab:blue')
# plt.plot(x, y_val, label='testing acc', color='tab:orange')
# plt.plot(x, y_adv_train, label='Adv training acc', color='tab:blue', linestyle='dashed')
# plt.plot(x, y_adv_val, label='Adv testing acc', color='tab:orange', linestyle='dashed')

# plt.title('Classification performance on Very Restricted Imagenet')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')

# plt.legend(loc = 'upper left')

# plt.show()
# plt.savefig('plot.png')
# plt.close()