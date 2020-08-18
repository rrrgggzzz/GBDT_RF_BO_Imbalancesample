import pandas as pd
import numpy as np
from imblearn.over_sampling import BorderlineSMOTE  # import BorderlineSMOTE
from pandas import DataFrame

# input data file
from sklearn.utils import compute_class_weight


df = pd.read_csv('F:\\undersampling.csv', header=None, sep=',',
                 names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC', 'class'])

target = 'class'
data = 'fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan', 'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'

x_columns = [x for x in df.columns if x not in [target]]
x = df[x_columns]
y = df['class']
groupby_data_orgianl = df.groupby('class').count()  # Classified summary of "class"
print(groupby_data_orgianl)  # print the classification distribution of the original sample set

# Use BorderlineSMOTE to oversample
model_bsmote = BorderlineSMOTE() #  build BorderlineSMOTE object
x_bsmote_resampled, y_bsmote_resampled = model_bsmote.fit_sample(x,y) # input data to oversample
x_bsmote_resampled = pd.DataFrame(x_bsmote_resampled, columns=['fault', 'road', 'river', 'lithology', 'elevation', 'slope',
                                                       'NDVI', 'profile', 'plan', 'aspect', 'geological', 'rain', 'SPI',
                                                       'TWI', 'TRI', 'STI', 'LUCC']) 
y_bsmote_resampled = pd.DataFrame(y_bsmote_resampled,columns=['class']) 
bsmote_resampled = pd.concat([x_bsmote_resampled, y_bsmote_resampled],axis=1) 
groupby_data_bsmote = bsmote_resampled.groupby('class').count() #Classified summary of "class"
print (groupby_data_bsmote) # Print the sample classification distribution of the output dataset processed by BorderlineSMOTE
exp = DataFrame(bsmote_resampled)
exp.to_csv("F://BSMOTE.csv") # output results
