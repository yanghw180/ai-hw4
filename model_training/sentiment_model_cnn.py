"""
Model definition for CNN sentiment training


"""

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import sequence    #
from tensorflow import keras
from tensorflow.keras.models import Sequential   #
from tensorflow.keras.layers import Embedding   #
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution1D, GlobalMaxPooling1D




def keras_model_fn(_, config):
    """
    Creating a CNN model for sentiment modeling

    """

    '''
    layer1: embedding layer
    '''
    input_length = config["padding_size"]
    input_dim = config["embeddings_dictionary_size"]
    output_dim = config["embeddings_vector_size"]

    f = open(config["embeddings_path"],encoding='utf8')
    glove = f.readlines()[:input_dim]
    f.close() 
    
    embedding_matrix = np.zeros((input_dim, output_dim))
    for i in range(input_dim):
        if len(glove[i].split()[1:]) != output_dim:
            continue
        embedding_matrix[i] = np.asarray(glove[i].split()[1:], dtype='float32')


    embedding_layer = Embedding(input_dim, output_dim, weights=[embedding_matrix],input_length=input_length, trainable=True)

    cnn_model = Sequential() 
    cnn_model.add(embedding_layer)


    """
    Layer2: Convolution1D layer
    """
    cnn_model.add(Convolution1D(
        filters=100,
        kernel_size=2,
        input_shape=(input_length, input_dim),
        strides=1,
        padding='valid',
        activation='relu'
    ))

    """
        Layer3: GlobalMaxPool1D layer
        """
    cnn_model.add(GlobalMaxPooling1D())
    """
    Layer4: Dense layer
    """
    cnn_model.add(Dense(units=100, activation='relu'))
    """
    Layer5: Dense layer
    """
    cnn_model.add(Dense(units=1, activation='sigmoid'))
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #cnn_model.compile(loss='binary_cprssentropy', opitmizer='adam', metrics=['accuracy'])
    cnn_model.compile('adam', loss='binary_crossentropy',  metrics=['accuracy'])
    return cnn_model

def save_model(model, output):
    """
    Method to save models in SaveModel format with signature to allow for serving
    """

    tf.saved_model.save(model, os.path.join(output, "1"))

    print("Model successfully saved at: {}".format(output))
