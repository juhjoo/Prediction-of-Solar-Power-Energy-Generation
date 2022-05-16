import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from src.data_generation import denormalize_label

colors = ['limegreen', 'violet', 'dodgerblue']

def predict_and_plot(models, input_data, target_data, sample_ind, future_target, norm_params, enc_tail_len=168):

    history_series = denormalize_label(input_data[:, :, -1], norm_params)    
    target_series = denormalize_label(target_data, norm_params)
    
    history_series = history_series[sample_ind:sample_ind+1,:]
    target_series = target_series[sample_ind:sample_ind+1,:] 
    
    history_series = history_series.reshape(-1,1)
    target_series = target_series.reshape(-1, 1)  

    history_series_tail = np.concatenate([history_series[-enc_tail_len:],target_series[:1]])
    x_history = history_series_tail.shape[0]
    plt.figure(figsize=(25, 12))  # (10,6) 
    plt.plot(range(1,x_history+1),history_series_tail)
    plt.plot(range(x_history,x_history+future_target),target_series,color='orange')
    
    for i in range(0, len(models)) :   
        forecast = models[i].predict(input_data)
        forecast = denormalize_label(forecast, norm_params)
        forecast = forecast[sample_ind:sample_ind+1,:]
        forecast = forecast.reshape(-1,1)   
        
        plt.plot(range(x_history,x_history+future_target),forecast,color=colors[i],linestyle='--')

    plt.title('Historic Series Tail , Target Series, and Predictions')
    plt.legend(['Historic Series','Target Series','Predictions(Proposed Model)', 'Predictions(DCCNN)', 'Predictions(LSTM)'])
    plt.xlabel('TIMESTEP')
    plt.ylabel('Actual')
    