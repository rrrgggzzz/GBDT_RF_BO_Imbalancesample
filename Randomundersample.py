import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler  # import RandomUnderSampler
from pandas import DataFrame

# input data file
from sklearn.utils import compute_class_weight

df = pd.read_csv('F:\\sample.csv', header=None, sep=',',
                 names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC', 'class'])

target = 'class'
data = 'fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan', 'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'

x_columns = [x for x in df.columns if x not in [target]]
x = df[x_columns]
y = df['class']
groupby_data_orgianl = df.groupby('class').count()  # Classified summary of "class"
print(groupby_data_orgianl)  # print the classification distribution of the original sample set

# Use RandomUnderSampler to undersample
model_RandomUnderSampler = RandomUnderSampler()  # build RandomUnderSampler object
x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled = model_RandomUnderSampler.fit_sample(x, y)  # input data to undersample
x_RandomUnderSampler_resampled = pd.DataFrame(x_RandomUnderSampler_resampled,
                                              columns=['fault', 'road', 'river', 'lithology', 'elevation', 'slope',
                                                       'NDVI', 'profile', 'plan', 'aspect', 'geological', 'rain', 'SPI',
                                                       'TWI', 'TRI', 'STI', 'LUCC'])
y_RandomUnderSampler_resampled = pd.DataFrame(y_RandomUnderSampler_resampled, columns=['class'])  
RandomUnderSampler_resampled = pd.concat([x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled], axis=1)  
groupby_data_RandomUnderSampler = RandomUnderSampler_resampled.groupby('class').count()  #Classified summary of "class"
print(groupby_data_RandomUnderSampler)  # Print the sample classification distribution of the output dataset processed by RandomUnderSampler
Random_sample = DataFrame(RandomUnderSampler_resampled)
Random_sample.to_csv("F://underSampling.csv") # output results