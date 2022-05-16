import pandas as pd
import numpy as np

def read_dataset(filename) :
    """
    Function for reading solar energy actual dataset
    : filename: solar energy actual dataset 
    : return: data
    """
    # Get time series data
    data = pd.read_csv(filename)
    
    # Change negative data.
    data.loc[data[(data['GB_UKM_solar_generation_actual'] < 0)]['GB_UKM_solar_generation_actual'].index, 'GB_UKM_solar_generation_actual'] = np.nan
    
    #feature selection
    data = data[['utc_timestamp','GB_UKM_solar_generation_actual']]

    data['utc_timestamp'] = pd.to_datetime(data['utc_timestamp'])
    data['utc_timestamp'] = data['utc_timestamp'] .dt.strftime('%Y-%m-%d %H:%M:%S')
    data['date'] = data['utc_timestamp'].str.split(' ').str[0]
    data['hour'] = data['utc_timestamp'].str.split(' ').str[1].str.split(':').str[0].astype(int)
    data['year'] = data['date'].str.split('-').str[0].astype(int)
    data['month'] = data['date'].str.split('-').str[1].astype(int)
    data['day'] = data['date'].str.split('-').str[2].astype(int)
    
    # preprocess n/a data using interpolate
    data.interpolate(method='values',order = 2,inplace=True, limit_direction='both')

    data = data[(data['utc_timestamp'] >= '2015-01-01 00:00:00') & (data['utc_timestamp'] < '2020-01-01 00:00:00')]
    
    #generate new feature(day sin and cos)
    day_sin = np.sin(data["hour"] * (2 * np.pi / 24)) / 2 + 0.5 
    day_cos = np.cos(data["hour"] * (2 * np.pi / 24)) / 2 + 0.5
    data.insert(loc = 0, column = "DAY_SIN", value = day_sin)
    data.insert(loc = 1, column = "DAY_COS", value = day_cos)
    
    return data

def read_weather_dataset(filename) :
    """
    Function for reading weather dataset
    :filename: weather dataset 
    :return: data
    """
    
    # Get time series data
    weather = pd.read_csv(filename)
    
    weather['utc_timestamp'] = pd.to_datetime(weather['utc_timestamp'])
    weather['utc_timestamp'] = weather['utc_timestamp'] .dt.strftime('%Y-%m-%d %H:%M:%S')
    
    #feature selection
    weather = weather[['utc_timestamp', 'GB_temperature', 'GB_radiation_direct_horizontal', 'GB_radiation_diffuse_horizontal']]
    weather = weather[(weather['utc_timestamp'] >= '2015-01-01 00:00:00') & (weather['utc_timestamp'] < '2020-01-01 00:00:00')]
    
    return weather

def preprocess_dataset(data, weather, past_history, future_target, STEP=1, validation_size=0.2) : 
    """
    Function for preprocessing dataset 
    :data : solar energy actual data
    :weather : weather data 
    :past_history : size of input
    :future_target : size of output
    :step: sampling of time series
    :validation_size : size of validation for time series
    :return: x_train, y_train, x_val, y_val, x_test, y_test, norm_params
    """
    
    ## merge a time series dataset
    df = weather.merge(data, left_on='utc_timestamp', right_on='utc_timestamp', how='outer')
    df = df[['year','GB_temperature', 'GB_radiation_direct_horizontal', 'GB_radiation_diffuse_horizontal', 'GB_UKM_solar_generation_actual', 'DAY_SIN', 'DAY_COS']]
    
    ## split a time series dataset(train, test)
    df_train = df[~(df['year'] == 2019)].reset_index(drop=True)
    df_test = df[(df['year'] == 2019)].reset_index(drop=True)
    df_train = df_train[['GB_temperature', 'GB_radiation_direct_horizontal', 'GB_radiation_diffuse_horizontal', 'DAY_SIN', 'DAY_COS', 'GB_UKM_solar_generation_actual']]
    df_test = df_test[['GB_temperature', 'GB_radiation_direct_horizontal', 'GB_radiation_diffuse_horizontal', 'DAY_SIN', 'DAY_COS', 'GB_UKM_solar_generation_actual']]
    TRAIN_SPLIT = int(df_train.shape[0] * (1 - validation_size))
    
    ## normalize time seires dataset
    norm_params = get_normalization_params(df_train[:TRAIN_SPLIT])
    df_train = normalize(df_train, norm_params)
    df_test = normalize(df_test, norm_params)
    
    df_train = np.array(df_train)
    df_test = np.array(df_test)
    
    ## Get x and y dataset for training and validation and testing
    x_train, y_train = multivariate_data(df_train, df_train[:, -1], 0, TRAIN_SPLIT, past_history, future_target, STEP, single_step=False)
    x_val, y_val = multivariate_data(df_train, df_train[:, -1], TRAIN_SPLIT, None, past_history, future_target, STEP, single_step=False)
    x_test, y_test = multivariate_data(df_test, df_test[:, -1], 0, None, past_history, future_target, STEP, single_step=False)
        
    return x_train, y_train, x_val, y_val, x_test, y_test, norm_params
 
def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False) :
    
    """
    Function for getting input and output window from dataset
    :dataset : x dataset
    :target : y data 
    :start_index : first x value
    :end_index :first y value
    :history_size : size of input indow
    :target_size : size of prediction window
    :step: sampling of time series
    :single_step : single or multi-step window
    :return: np.array(data), np.array(labels)
    """    

    data = []
    labels = []
    
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])
    return np.array(data), np.array(labels)
   
def get_normalization_params(data):
    """
    Function for getting parameters from dataset
    :data : time seires dataset
    :return: norm_params
    """  
    norm_params = {}
    norm_params['mean'] = data.mean()
    norm_params['std'] = data.std()
    norm_params['max'] = data.max()
    norm_params['min'] = data.min()

    return norm_params

def normalize(data, norm_params, method='minmax'):
    """
    Function for normalizing dataset
    :data : time seires dataset
    :norm_params : mean, std, max, min 
    :method : zscore or minmax
    :return: data
    """    
    if method == 'zscore':
        return (data - norm_params['mean']) / norm_params['std']
    elif method == 'minmax':
        return (data - norm_params['min']) / (norm_params['max'] - norm_params['min'])
    elif method is None:
        return data
    
def denormalize(data, norm_params, method='minmax'):
    """
    Function for denormalizing dataset
    :data : time seires dataset
    :norm_params : mean, std, max, min 
    :method : zscore or minmax
    :return: data
    """    
    if method == 'zscore':
        return (data * norm_params['std']) + norm_params['mean']
    elif method == 'minmax':
        return (data * ((norm_params['max']) - (norm_params['min'])) + norm_params['min'])
    elif method is None:
        return data

def denormalize_label(data, norm_params, method='minmax'):
    """
    Function for denormalizing label in dataset
    :data : time seires dataset
    :norm_params : mean, std, max, min 
    :method : zscore or minmax
    :return: data
    """    
    if method == 'zscore':
        return (data * norm_params['std']['GB_UKM_solar_generation_actual']) + norm_params['mean']['GB_UKM_solar_generation_actual']
    elif method == 'minmax':
        return (data * ((norm_params['max']['GB_UKM_solar_generation_actual']) - (norm_params['min']['GB_UKM_solar_generation_actual'])) + norm_params['min']['GB_UKM_solar_generation_actual'])
    elif method is None:
        return data