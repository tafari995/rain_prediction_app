from EncDecDataclass import *
from TS_TransformerRainbrain_2 import *


def rpred(
        hourly_df, 
        back_window, 
        hours_forward, 
        batch_size 
    ) -> np.array:
    hourly_df = hourly_df.to_numpy()
    my_data = MyDataset(hourly_df)
    my_data.sm_win = back_window
    b_sz = batch_size

    brain = TSTransformer(
        n_features = my_data.data.shape[1]*my_data.sm_win, 
        num_heads = 1
    )
    my_data.window()
    my_data.remove_trend()
    my_data.scale_data() 
    data_loader = my_data.batch_n_load(batch_size=b_sz)   
    brain.__train__(data_loader, 60, learning_rate=0.002) 

    prd = predict_stream(
        brain, 
        my_data.data[- b_sz:-1,:], 
        hrs = hours_forward
    )
    prd = MyDataset(prd.numpy())
    prd.scaler = my_data.scaler
    prd.trend_col = my_data.trend_col
    prd.scale_data(inverse=True)
    prd.remove_trend(inverse=True)
    prd.sm_win = my_data.sm_win
    prd.window(inverse=True)
    prediction = prd.data[:,-1].numpy()
    return prediction
