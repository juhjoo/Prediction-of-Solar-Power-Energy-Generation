from src.experience import run_model

DATA_FILE_NAME = 'data/time_series_60min_singleindex.csv'
WEATHER_FILE_NAME = 'data/weather_data.csv'

DCLSTM_RESULT_FILE_NAME = 'result/result_dclstm.csv'
DCCNN_RESULT_FILE_NAME = 'result/result_dccnn.csv'
LSTM_RESULT_FILE_NAME = 'result/result_lstm.csv'

FUTURE_TARGET = 24
PAST_HISTORY = [168]
BATCH_SIZE = [256]
EPOCHS = [50]
BUFFER_SIZE = 10000
METRICS = ['mse','rmse','nrmse','mae','wape']


dclstm_params = {
    'n_filters': [128],
    'filter_length': [2, 3, 4],
    'n_blocks': [3, 4],
    'dilation_rate': [[1, 2, 4, 8, 16], [1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32, 64]],
    'units': [32, 64, 128],
    'num_layers': [1],
    'dropout': [0]}
  
dccnn_params = {
  'n_filters': [128],
  'filter_length': [2, 3, 4],
  'n_blocks': [3, 4],
  'dilation_rate': [[1, 2, 4, 8, 16], [1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32, 64]]}

lstm_params = {
  'num_stack_layers': [1, 2, 3],
  'units': [32, 64, 128],
  'dropout': [0]
  }

run_model(data_name=DATA_FILE_NAME, 
          weather_name=WEATHER_FILE_NAME, 
          result_file=DCLSTM_RESULT_FILE_NAME, 
          future_target_in=FUTURE_TARGET, 
          past_history_in=PAST_HISTORY, 
          batch_size_in=BATCH_SIZE, 
          epochs_in=EPOCHS, 
          params=dclstm_params, 
          model='dclstm', 
          metric_in=METRICS, 
          buffer_size=10000, 
          validation_size=0.2, 
          seed=1)

run_model(data_name=DATA_FILE_NAME, 
          weather_name=WEATHER_FILE_NAME, 
          result_file=DCCNN_RESULT_FILE_NAME, 
          future_target_in=FUTURE_TARGET, 
          past_history_in=PAST_HISTORY, 
          batch_size_in=BATCH_SIZE, 
          epochs_in=EPOCHS, 
          params=dccnn_params, 
          model='dccnn', 
          metric_in=METRICS, 
          buffer_size=10000, 
          validation_size=0.2, 
          seed=1)

run_model(data_name=DATA_FILE_NAME, 
          weather_name=WEATHER_FILE_NAME, 
          result_file=LSTM_RESULT_FILE_NAME, 
          future_target_in=FUTURE_TARGET, 
          past_history_in=PAST_HISTORY, 
          batch_size_in=BATCH_SIZE, 
          epochs_in=EPOCHS, 
          params=lstm_params, 
          model='lstm', 
          metric_in=METRICS, 
          buffer_size=10000, 
          validation_size=0.2, 
          seed=1)

