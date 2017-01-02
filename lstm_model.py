from __future__ import print_function
from keras.models import Graph
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization

"""
LSTM params
Structure taken from RoboTrump LSTM repo
"""

input_dim = 300
lstm_hdim = 500
bridge_dim = 1000
dense_dim = 1500
sd_len = 12
batch_size = 256

layerNames = ['tdd1', 'bn1', 'lstm1', 'bn2', 'lstm2', 'bn3', 'lstm3', 'bn4', 'dropout1', 'dense1', 'denseelu1',
              'dropout2', 'dense2','densesm1']


def create_model(word_coding):
    """
    Create the LSTM model
    :param word_coding:
    :return:
    """
    model = Graph()
    model.add_input(name='input', input_shape=(sd_len, input_dim))
    model.add_node(TimeDistributedDense(input_dim=input_dim, output_dim=lstm_hdim, input_length=sd_len),
                   name=layerNames[0], input='input')
    model.add_node(BatchNormalization(), name=layerNames[1], input=layerNames[0])

    model.add_node(LSTM(input_dim=lstm_hdim, output_dim=lstm_hdim, return_sequences=True), name=layerNames[2] + 'left',
                   input=layerNames[1])
    model.add_node(BatchNormalization(), name=layerNames[3] + 'left', input=layerNames[2] + 'left')

    model.add_node(LSTM(input_dim=lstm_hdim, output_dim=lstm_hdim, return_sequences=True, go_backwards=True),
                   name=layerNames[2] + 'right', input=layerNames[1])
    model.add_node(BatchNormalization(), name=layerNames[3] + 'right', input=layerNames[2] + 'right')

    model.add_node(LSTM(input_dim=lstm_hdim, output_dim=lstm_hdim, return_sequences=False), name=layerNames[6] + 'left',
                   input=layerNames[3] + 'left')

    model.add_node(LSTM(input_dim=lstm_hdim, output_dim=lstm_hdim, return_sequences=False, go_backwards=True),
                   name=layerNames[6] + 'right', input=layerNames[3] + 'right')

    model.add_node(BatchNormalization(), name=layerNames[7], inputs=[layerNames[6] + 'left', layerNames[6] + 'right'])
    model.add_node(Dropout(0.2), name=layerNames[8], input=layerNames[7])

    model.add_node(Dense(input_dim=bridge_dim, output_dim=dense_dim), name=layerNames[9], input=layerNames[8])
    model.add_node(ELU(), name=layerNames[10], input=layerNames[9])
    model.add_node(Dropout(0.2), name=layerNames[11], input=layerNames[10])

    model.add_node(Dense(input_dim=dense_dim, output_dim=len(word_coding)), name=layerNames[12], input=layerNames[11])
    model.add_node(Activation('softmax'), name=layerNames[13], input=layerNames[12])
    model.add_output(name='output1', input=layerNames[13])

    model.compile(optimizer='rmsprop', loss={'output1': 'categorical_crossentropy'})

    return model

