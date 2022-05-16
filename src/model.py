from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Flatten, TimeDistributed, Conv1D, Input, Dense, Multiply, Add, Activation
from tensorflow.keras import Model, Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers

def DC_CNN_Block(z, nb_filter, filter_length, dilation, l2_layer_reg) :
    """
    Function of DCCNN Block
    :z: layer 
    :nb_filter: size of filters 
    :filter_length: size of kernel
    :dilation: dilation rate
    :l2_layer_reg: l2 regulation 
    :return: network_out, skip_out
    """
    residual = z

    layer_out = Conv1D(filters=nb_filter, kernel_size=filter_length, dilation_rate=dilation, activation='linear',
                     padding='causal', use_bias=False, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,seed=42), 
                     kernel_regularizer=l2(l2_layer_reg))(z)
                   
    layer_out = Activation('relu')(layer_out)
        
    skip_out =  Conv1D(1,1, activation='linear', use_bias=False, 
                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, 
                    seed=42), kernel_regularizer=l2(l2_layer_reg))(layer_out)
        
    network_in = Conv1D(1,1, activation='linear', use_bias=False, 
                      kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, 
                      seed=42), kernel_regularizer=l2(l2_layer_reg))(layer_out)
                      
    network_out = Add()([residual, network_in])
        
    return network_out, skip_out


def DC_CNN_LSTM_Model(input_shape, output_size, n_filters, filter_length, n_blocks, dilation_rates, units = 124, num_layers=1, dropout=0):
    """
    Function of DCLSTM(Proposed model) Model
    :input_shape: size of input 
    :output_size: size of output 
    :nb_filter: size of filters 
    :filter_length: size of kernel
    :n_blocks: size of block
    :dilation_rates: dilation rate 
    :units: size of LSTM unit
    :num_layers: size of LSTM layer
    :return: model
    """
    inputs = keras.Input(shape=(input_shape))
    z = inputs
    skip_list = []
    for dilation_rate in dilation_rates * n_blocks:
      z, skip = DC_CNN_Block(z, n_filters, filter_length, dilation_rate, 0.001)
      if dilation_rate >= 32:
        skip = tf.keras.layers.Dropout(0.8)(skip)
      skip_list.append(skip)
    z = Add()(skip_list)
    z = Activation('relu')(z)
    z =  Conv1D(1,1, activation='linear', use_bias=False, 
                  kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=42),
                  kernel_regularizer=l2(0.001))(z)

    return_sequences = num_layers>1
    y = tf.keras.layers.LSTM(units, return_sequences=return_sequences, dropout=dropout)(inputs)
    for i in range(num_layers-1):
      return_sequences = i<num_layers-2
      x = tf.keras.layers.LSTM(units, return_sequences=return_sequences, dropout=dropout)(y)
    y = keras.layers.Dense(output_size)(y)
    z = keras.layers.Flatten()(z)
    z = keras.layers.Dense(output_size)(z)
    z = Multiply()([y, z])
    z = Activation('relu')(z)
    z = tf.keras.layers.Dropout(0.1)(z)
    out = keras.layers.Dense(output_size)(z)
    model = keras.Model(inputs=[inputs], outputs=[out])

    return model

def DC_CNN_Model(input_shape, output_size, n_filters, filter_length, n_blocks, dilation_rates):
    """
    Function of DCCNN Model
    :input_shape: size of input 
    :output_size: size of output 
    :nb_filter: size of filters 
    :filter_length: size of kernel
    :n_blocks: size of block
    :dilation_rates: dilation rate 
    :return: model
    """
    
    inputs = keras.Input(shape=(input_shape))
    z = inputs
    skip_list = []
    for dilation_rate in dilation_rates * n_blocks:
      z, skip = DC_CNN_Block(z, n_filters, filter_length, dilation_rate, 0.001)
      if dilation_rate >= 32:
        skip = tf.keras.layers.Dropout(0.8)(skip)
      skip_list.append(skip)
    z =   Add()(skip_list)

    z =   Activation('relu')(z)

    z =  Conv1D(1,1, activation='linear', use_bias=False, 
                  kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=42),
                  kernel_regularizer=l2(0.001))(z)
    z = keras.layers.Flatten()(z)
    z = keras.layers.Dense(output_size)(z)
    z = Activation('relu')(z)
    z = tf.keras.layers.Dropout(0.1)(z) 
    out = keras.layers.Dense(output_size)(z)

    model = keras.Model(inputs=[inputs], outputs=[out])

    return model


def lstm_model(input_shape, output_size=1, num_stack_layers=1, units=50, dropout=0):
    """
    Function of LSTM Model
    :input_shape: size of input 
    :output_size: size of output  
    :num_stack_layers: size of LSTM layer
    :units: size of LSTM unit
    :dropout: size of LSTM dropout
    :return: model
    """
        
    inputs = tf.keras.layers.Input(shape=input_shape)
    return_sequences = num_stack_layers>1
    z = tf.keras.layers.LSTM(units, return_sequences=return_sequences, dropout=dropout)(inputs)
    for i in range(num_stack_layers-1):
        return_sequences = i<num_stack_layers-2
        z = tf.keras.layers.LSTM(units, return_sequences=return_sequences, dropout=dropout)(z)
    z = tf.keras.layers.Dense(output_size)(z)

    model = tf.keras.Model(inputs=inputs, outputs=z)
    
    return model