from src.data_generation import read_dataset, read_weather_dataset, preprocess_dataset, denormalize_label
from src.model import DC_CNN_Model, DC_CNN_LSTM_Model, lstm_model
from src.metric import evaluate, evaluate_all
import itertools
import time
import os
import tensorflow as tf
import numpy as np



def run_model(data_name, weather_name, result_file, future_target_in, past_history_in, batch_size_in, epochs_in, params, model, metric_in, buffer_size=10000, validation_size=0.2, seed=1) :
  
    checkpoint_path = "training/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    ## Read dataset
    data = read_dataset(data_name)
    weather = read_weather_dataset(weather_name)

    ## Write result file
    current_index = 0
    if current_index == 0 :
      with open(result_file, 'w') as resfile :
        resfile.write(';'.join([str(a) for a in ['MODEL', 'MODEL_DESCRIPTION', 'FORECAST_HORIZON', 'PAST_HISTORY', 'BATCH_SIZE', 'EPOCHS'] + metric_in + ['val_' + m for m in metric_in] + ['loss', 'val_loss']]) + "\n")
    
    for past_history, batch_size, epochs in list(itertools.product(past_history_in, batch_size_in, epochs_in)) :  
      ## Preprocess time series dataset and get x, y for train, test and validation
      x_train, y_train, x_val, y_val, x_test, y_test, norm_params = preprocess_dataset(data, weather, past_history, future_target_in)
    
      train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).cache().shuffle(buffer_size).batch(batch_size).repeat()
      val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).repeat()
      test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    
      ## Create models
      model_list = {}

      if model == 'lstm' :
          model_list = {'lstm_Model_{}'.format(j) : (lstm_model, [x_train.shape[-2:], future_target_in, *params]) for j, params in enumerate(itertools.product(*params.values()))}
      
      if model == 'dccnn' :
          model_list = {'DCCNN_Model_{}'.format(j) : (DC_CNN_Model, [x_train.shape[-2:], future_target_in, *params]) for j, params in enumerate(itertools.product(*params.values())) if params[1] * params[2] * params[3][-1] >= past_history}
           
      if model == 'dclstm' : 
          model_list = {'DC_CNN_LSTM_Model{}'.format(j) : (DC_CNN_LSTM_Model, [x_train.shape[-2:], future_target_in, *params]) for j, params in enumerate(itertools.product(*params.values())) if params[1] * params[2] * params[3][-1] >= past_history}
      
      steps_per_epoch = int(np.ceil(x_train.shape[0] / batch_size))
      validation_steps = steps_per_epoch if val_data else None
    
      for model_name, (model_function, params) in model_list.items() :
        model = model_function(*params)
        print(model.summary())
        model.compile(loss='mae', optimizer='adam', metrics=['mse']) 
        print(*params)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, monitor='val_loss', save_best_only='True')
        
        ## Train model
        history = model.fit(train_data, epochs=epochs, validation_data=val_data, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, callbacks=[cp_callback])
        model.load_weights(checkpoint_path)
        model.save('model/'+ model_name + '.h5')
        val_metrics = {}
    
        ## Get validation result metrics
        if validation_size > 0 :
          val_forecast = model.predict(x_val)
          val_forecast = denormalize_label(val_forecast, norm_params)
          y_val_denorm = denormalize_label(y_val, norm_params)
          x_val_denorm = denormalize_label(x_val[:, :, -1], norm_params)    
            
          val_metrics = evaluate(y_val_denorm, val_forecast, metric_in)
          print('Val metrics : ', val_metrics)
        
        ## Get test result metrics
        test_forecast = model.predict(test_data)
        test_forecast = denormalize_label(test_forecast, norm_params)
        y_test_denorm = denormalize_label(y_test, norm_params)
        x_test_denorm = denormalize_label(x_test[:, :, -1], norm_params)
          
        test_metrics = evaluate(y_test_denorm, test_forecast, metric_in)
        print('Test metrics : ', test_metrics)
    
        val_metrics = {'val_' + k: val_metrics[k] for k in val_metrics}
        print('Val metrics : ', val_metrics)
        
        ## Save a result
        model_metric = {'MODEL' : model_name,
                        'MODEL_DESCRIPTION' : params,
                        'FORECAST_HORIZON' : future_target_in,
                        'PAST_HISTORY' : past_history,
                        'BATCH_SIZE' : batch_size,
                        'EPOCHS' : epochs,
                        **test_metrics,
                        **val_metrics,
                        'loss' : history.history['loss'],
                        'var_loss' : history.history['val_loss'],    
                        }
        
        ## Write a result file
        with open(result_file, 'a') as resfile :
          resfile.write(';'.join([str(a) for a in model_metric.values()]) + "\n")

    
  