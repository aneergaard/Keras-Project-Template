import datetime

import numpy as np
from keras.initializers import Zeros
from keras.initializers import glorot_uniform
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Reshape
from keras.layers.cudnn_recurrent import CuDNNLSTM
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import multi_gpu_model
from keras.utils import print_summary

from base.base_model import BaseModel
from utils.config import process_config


class CustomModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.CNN_window_size = config.model.CNN_window_size
        self.dense_units = {'DenseLayer1': 128,
                            'DenseLayer2': 128}  # {k: np.random.randint(v[0], v[1]) for k,v in config.dense_units.items()}
        self.dimensionality = config.model.dimensionality
        self.dropout_probability = config.model.dropout_probability
        self.input_shape = config.model.input_shape
        self.kernel_size = config.model.kernel_size
        self.initial_lr = config.model.initial_lr
        self.num_gpus = config.model.num_gpus
        self.num_LSTM_units = config.model.num_LSTM_units
        self.num_time_steps = config.model.num_time_steps
        self.padding = config.model.padding
        self.parallel_model = None
        self.scale_l2_regularization = config.model.scale_l2_regularization
        self.stride = config.model.stride
        self.x = None
        self.y = None
        self.build_model()

    def build_model(self):
        self.x = Input(shape=self.input_shape)
        self.y = self.x

        # CNN layers
        for layer in ['Layer1', 'Layer2']:
            self.y = Conv2D(filters=self.dimensionality[layer],
                            kernel_size=(1, self.kernel_size[layer]),
                            padding=self.padding[layer],
                            data_format='channels_last',
                            activation='relu',
                            kernel_initializer=glorot_uniform(),
                            bias_initializer=Zeros())(self.y)
            self.y = BatchNormalization()(self.y)

        # WidthLayer2
        if self.padding['Layer2'] == 'same':
            WidthLayer2 = self.CNN_window_size
        else:
            WidthLayer2 = np.int(
                np.floor((self.CNN_window_size - self.kernel_size['Layer2']) / self.stride['Layer2']) + 1)

        # 1st Dense layer
        self.y = Reshape((self.num_time_steps, self.dimensionality['Layer2'] * WidthLayer2))(self.y)
        self.y = Dense(units=self.dense_units['DenseLayer1'],
                       activation='relu',
                       kernel_regularizer=l2(self.scale_l2_regularization))(self.y)
        self.y = Dropout(rate=1 - self.dropout_probability)(self.y)
        self.y = BatchNormalization()(self.y)

        # LSTM Layer
        self.y = CuDNNLSTM(self.num_LSTM_units, return_sequences=True)(self.y)

        # 2nd Dense layer
        self.y = Dense(units=self.dense_units['DenseLayer2'],
                       activation='relu',
                       kernel_regularizer=l2(self.scale_l2_regularization))(self.y)
        self.y = Dropout(rate=1 - self.dropout_probability)(self.y)
        self.y = BatchNormalization()(self.y)

        # Softmax
        self.y = Dense(units=1, activation='sigmoid')(self.y)

        self.model = Model(inputs=self.x, outputs=self.y, name='model')

        if self.num_gpus > 1:
            print('{} | [INFO] | Training with {} GPUs '.format(datetime.datetime.now(), self.num_gpus))
            self.parallel_model = multi_gpu_model(self.model, gpus=self.num_gpus, cpu_merge=False)
            self.parallel_model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                                        optimizer=Adam(lr=self.initial_lr))
        else:
            self.model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                               optimizer=Adam(lr=self.initial_lr), sample_weight_mode='temporal')


if __name__ == '__main__':
    config = process_config('./configs/test.json')
    model = CustomModel(config)
    print_summary(model.model)
