import tensorflow as tf
from tensorflow import keras
from keras import layers, Input, Model, callbacks, activations, regularizers
from tensorflow.keras.utils import Sequence
from keras.utils.vis_utils import plot_model
import os
import numpy as np

def MLP(input_shape,dropout_rate,learning_rate):
    
    inputs = Input(shape=input_shape)
    x      = layers.Dense( 500, activation='relu')(inputs)
    x      = layers.Dropout(dropout_rate)(x)
    x      = layers.Dense( 300, activation='relu')(x)
    x      = layers.Dropout(dropout_rate)(x)
    x      = layers.Dense( 150, activation='relu')(x)
    x      = layers.Dropout(dropout_rate)(x)
    x      = layers.Dense( 50 , activation='relu')(x)
    x      = layers.Dropout(dropout_rate)(x)
    x      = layers.Dense( 50 , activation='relu')(x)
    outs   = layers.Dense(1, activation='sigmoid')(x)
        
    model=Model(inputs, outs)
    optimizer   = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def Train_mlp(X_train, X_val, y_train, y_val,input_shape,L2_regularization,learning_rate):
    
    model=MLP(input_shape,L2_regularization,learning_rate)
    reduce_LR = callbacks.ReduceLROnPlateau(monitor='val_loss',patience=2, verbose=1, factor=0.5, min_lr=0.000000001)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=250,restore_best_weights=True,verbose=0)

    weights_file = './Models/space_mlp.h5'

    if os.path.exists(weights_file):
        model.load_weights(weights_file)
        my_history=np.load('./Models/my_history.npy',allow_pickle='TRUE').item()
        print('Loaded weights!')
    else:
        history_keras =  model.fit(X_train,y_train, epochs=1000, validation_data=(X_val,y_val), callbacks=[early_stopping],verbose=1)
        model.save_weights('space_mlp.h5')
        np.save('my_history.npy',history_keras.history)
        my_history=history_keras.history
    return model,my_history

def predict_mlp(model,test_x):
    return model.predict(test_x) 