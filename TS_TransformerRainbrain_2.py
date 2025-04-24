
import numpy as np
import torch
from torch import nn, zeros_like, cat, stack, triu, clone, full, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

def generate_square_subsequent_mask(
    sz: int,
    dtype: torch.dtype = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    return triu(
        full((sz, sz), float("-inf"), dtype=dtype),
        diagonal=1,
    )

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=None, max_len=1024):
        super(PositionalEncoding, self).__init__()
        if dropout:
            assert type(dropout)==float
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float)   #.unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(max_len) / d_model))
        for j in range(d_model):
            if j%2==0:
                pe[:,j] += torch.sin(position * np.exp(2*j*(-np.log(max_len) / d_model)) )
            else:
                pe[:,j] += torch.cos(position * np.exp(2*j*(-np.log(max_len) / d_model)) )
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        if self.dropout:
            return self.dropout(x)
        else:
            return x

class TSTransformer(nn.Module):
    def __init__(self, n_features, num_heads=1):
        super(TSTransformer, self).__init__()
        # Positional Encoding
        self.pe = PositionalEncoding(n_features)
        # Encoder
        base_enc_layer = TransformerEncoderLayer(
            d_model = n_features, 
            nhead = num_heads,
            dim_feedforward = 512,
            batch_first = True
        )
        self.encoder = TransformerEncoder(
            base_enc_layer, 
            num_layers = 1
        )        
        # Decoder
        base_dec_layer = TransformerDecoderLayer(
            d_model = n_features, 
            nhead = num_heads,
            dim_feedforward = 512,
            batch_first = True            
        )
        self.decoder = TransformerDecoder(
            base_dec_layer, 
            num_layers = 1
        )
        
        # Projections
        self.proj = nn.Linear(n_features, n_features)
        
        
    def forward(self, x, memory = None):
        # Input shape: [sequence_length, n_features ]        
        seq_len = x.size(0)
        # Positional Enc
        x = self.pe(x)     
        # Encoder
        memory = self.encoder(x)          
        # Decoder
        x_dec = self.decoder(
            x, 
            memory,
            tgt_mask = generate_square_subsequent_mask(seq_len)
        )
        
        # Final projection
        output = self.proj(x_dec)
        
        return output
        
    def __train__(self, dataloader, num_epochs=10, learning_rate=0.001):
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        N = len(dataloader)
        # Training loop
        self.train()
        for epoch in range(num_epochs):            
            total_loss = 0
            
            for batch in dataloader:
                x = batch  
                # zero gradients
                optimizer.zero_grad()                
                # forward pass
                outputs = self(x[:-1,:])
                loss = loss_fn(outputs, x[1:,:])                
                # backward pass and optimize
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # print statistics
            avg_loss = total_loss / N
            print(f'Epoch: {epoch+1} / {num_epochs} \n Average Loss: {avg_loss:.4f}')
        
        return self
    
    '''
    Since we want to train on windowed data, we must also predict
    on windowed data. Once prediction is complete, we can drop
    the lagged features for ease of visualization and analysis
    '''    
    
def predict_batch(model, dataloader): 
    model.eval()           
    predictions = []        
    with torch.no_grad():
        for batch in dataloader:
            pred = model(batch)
            predictions.append(pred)
    return stack(predictions)    
    
def predict_stream(model, data, hrs = 1): 
    model.eval()           
    predictions = []    
    
    assert type(hrs) == int
    with torch.no_grad():
        for _ in range(hrs):
            pred = model(data)
            predictions.append(pred[-1,-1,:])
            data = pred
    return stack(predictions)
            

# Example usage:
'''
model = TSTransformer(n_features = 5, num_heads = 1)
# Define input tensor with shape [batch_size, n_features, sequence_length]
input_tensor = torch.randn(5, 50, 5)  # batch_size=5, seq_len=50, n_features=5 
print("Model input:", input_tensor)
output = model(input_tensor)
print("Model output:", output)
'''

#code to test brain
'''
from EncDecDataclass import *
from TS_TransformerRainbrain_2 import *
from prediction_data_exampler import *
city = "Seattle"  
s_date = "2025-03-09"
e_date = "2025-04-19"
hourly_df = get_hourly_weather(city,s_date,e_date)

hourly_df = hourly_df.to_numpy()
my_data = MyDataset(hourly_df)
my_data.sm_win = 6
b_sz = 17

brain = TSTransformer(
    n_features = my_data.data.shape[1]*my_data.sm_win, 
    num_heads = 1
)
my_data.window()
my_data.remove_trend()
my_data.scale_data() 
data_loader = my_data.batch_n_load(batch_size=b_sz)   
brain.__train__(data_loader, 50, learning_rate=0.002) 

prd = predict_stream(brain, my_data.data[- b_sz:-1,:], hrs=72)
prd = MyDataset(prd.numpy())
prd.scaler = my_data.scaler
prd.trend_col = my_data.trend_col
prd.scale_data(inverse=True)
prd.remove_trend(inverse=True)
prd.sm_win = my_data.sm_win
prd.window(inverse=True)
prd.data


my_data.already_batched = False
my_data.data
data_loader2 = my_data.batch_n_load(batch_size=16)
brain.predict(data_loader2)
'''


'''

### Usage Example:
```python
# Create a test dataset (replace with your actual dataset)
def get_time_series_dataset(num_samples=100, n_features=5):
    # Generate random time series data
    tensor = torch.randn(num_samples, n_features, 50)  # [num_samples, n_features, sequence_length]
    
    # Split into train and test sets
    train_set = torch.utils.data.SubDataset(tensor, slice=None, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32)
    test_loader = torch.utils.data.DataLoader(
        tensor[len(train_set):], 
        batch_size=32,
        shuffle=False
    )
    
    return train_loader, test_loader

# Initialize model and train it
model = TimeSeriesTransformer(n_features=5).cuda()
train_loader, _ = get_time_series_dataset(num_samples=100)
model = train_model(model, train_loader, num_epochs=10, learning_rate=0.001)

# Save trained model
torch.save(model.state_dict(), "time_series_model.pth")

# Load model for prediction
model = TimeSeriesTransformer(n_features=5).cuda()
model.load_state_dict(torch.load("time_series_model.pth"))

# Generate a new batch of data for prediction (replace with your actual test data)
test_data = torch.randn(32, 5, 50).cuda()  # [batch_size, n_features, sequence_length]
preds = predict_model(model, iter([test_data]))  # preds shape: [batch_size, 1]
print(preds)

hourly_df = add_engineered_features(
    hourly_df,
    explicit_features = ["surface_pressure","wind_speed_10m","rain"],
    lags = [1,2,4,12]
)

'''





