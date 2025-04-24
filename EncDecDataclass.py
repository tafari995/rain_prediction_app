
import numpy as np
from sklearn.linear_model import LinearRegression
from numpy.random import default_rng
from torch import tensor, roll, cat, clone, min, max
from torch.utils.data import Dataset, DataLoader


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.data_min = None
        self.data_max = None

    def fit(self, data):
        self.vec_min = data.min(dim=0, keepdim=True)[0]
        self.vec_max = data.max(dim=0, keepdim=True)[0]

    def transform(self, data):
         if self.vec_max is None:
            raise RuntimeError("fit() must be called before transform()")
         data_range = (self.vec_max - self.vec_min)
         scaled_data = (data - self.vec_min) / data_range
         
         a, b = self.feature_range
         scaled_data = scaled_data * (b - a) + a
         return scaled_data

    def inverse_transform(self, data):
        if self.vec_max is None:
            raise RuntimeError("fit() must be called before inverse_transform()")        
        a, b = self.feature_range        
        data = (data - a) / (b - a)        
        original_data = data * (self.vec_max - self.vec_min) + self.vec_min
        return original_data

class MyDataset(Dataset):
    def __init__(
        self, 
        input_data,
        already_batched=False, 
        batch_size=None, 
        data_loader=None, 
        ):
        self.scaler = None
        self.trend_col = None
        self.sm_win = None
        
        # saving raw numpy array for ease and speed
        self.raw_data = input_data
                
        #  the last column of my input data contains rain
        self.rain = self.raw_data[:,self.raw_data.shape[1]-1]
         
        #  numpy arrays get converted to tensors for training/prediction
        self.data = tensor(input_data).float()    
               
        #  attributes of the dataset for processing    
        self.already_batched = already_batched
        self.data_loader = data_loader

    def __len__(self):
        return len(self.data)
        
    def shape(self):
        return self.data.shape
        
    def __getitem__(self, idx):
        return self.data[idx]
        
    def remove_trend(self, inverse = False):
        T_rows = self.data.shape[0]
        hrs_passed = np.arange(T_rows).reshape(-1,1)
        np_data = self.data.numpy()        
        arr = np_data
        if not inverse:
            trend_collector = np.zeros_like(arr)
            for c_i in range(arr.shape[1]):                
                regr_obj = LinearRegression(
                                 ).fit(hrs_passed, arr[:,c_i]
                                 )
                vert_pred = regr_obj.predict(hrs_passed).reshape(-1,)                 
                trend_collector[:,c_i] += vert_pred 
                arr[:,c_i] -= vert_pred
            self.trend_col = trend_collector
        elif inverse:
            for c_i in range(arr.shape[1]):
                arr[:,c_i] += self.trend_col[:arr.shape[0],c_i]                
        self.data = tensor(np_data).float()        
        
    def scale_data(self,inverse=False):
        if not inverse:
            scaler = MinMaxScaler(feature_range = (-1,1))
            scaler.fit(self.data)  
            self.data = scaler.transform(self.data)         
            self.scaler = scaler 
        elif inverse:
            scaler = self.scaler
            self.data = scaler.inverse_transform(self.data)           
                       
        
    def batch_n_load(self, batch_size):
        '''
        method to put data in a dataloader object
        with the specified batch size, and update the
        relevant dataset attributes
        '''
        assert self.already_batched == False         
                     
        self.data_loader = DataLoader(self, batch_size=batch_size)       
        self.already_batched = True
        self.batch_size = batch_size
        return self.data_loader

    def window(self, inverse = False):
        if not inverse:            
            small_win = self.sm_win 
            assert type(small_win) == int and small_win>-1
            roller = clone(self.data)
            for _ in range(1,small_win):                
                roller = roll(roller,1,0)
                roller[0] = roller[1]
                self.data = cat((self.data, roller),1)
        elif inverse:
            self.data = self.data[:, :(self.shape()[1]//self.sm_win) ]

'''
from EncDecDataclass import *
from prediction_data_exampler import *
city = "Seattle"  
s_date = "2025-02-10"
e_date = "2025-03-09"
hourly_df = get_hourly_weather(city,s_date,e_date)
hourly_df.dropna()

hourly_df = hourly_df.to_numpy()

my_data = MyDataset(hourly_df)
my_data.remove_trend()
my_data.scale_data()
my_data.data
my_data.shape
my_data.scale_data(inverse=True)
my_data.remove_trend(inverse=True)
my_data.data
my_data.sm_win = 6
for batch in my_data.batch_n_load(batch_size = 16):
    print(batch[0:,:])
'''
