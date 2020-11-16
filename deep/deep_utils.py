import logging
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Activation
import keras
import os
import multiprocessing as mp
import tensorflow as tf

class DeepBaseModel:
    def __init__(self, model_name = 'deep_model',
                 activation = 'sigmoid',
                 loss = 'binary_crossentropy',
                 optimizer = 'adam',
                 batch_size = 128,
                 dropout_rate = 0.4,
                 epochs = 50,
                 initial_learning_rate = 0.001,
                 vector_size = 300,
                 use_gpu= True):
        '''

        Args:
            model_name: str. choose a name for the model
            activation: str. activation function for the deep layers. for example, 'relu', 'sigmoid'.
            loss: str. loss function.
            optimizer: str. deep model optimizer. choose from ['adam', 'adadelta', 'adagrad', 'adamax']
            batch_size: int. batch size.
            dropout_rate: float. dropout layer rate
            epochs: int. number of epochs.
            initial_learning_rate:  float. initial learning model for the deep model.
            vector_size: int. embedding vector size.
            use_gpu: boolean [True, False]
        '''
        self.model_name = model_name
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size =batch_size
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.vector_size = vector_size
        self.initial_learning_rate = initial_learning_rate
        self.use_gpu = use_gpu
        self.logdir = f'deep_logs/{model_name}'

    def config_gpu(self):
        if self.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            config = tf.ConfigProto(device_count={"GPU": 1, "CPU": mp.cpu_count()}, log_device_placement=True)
            config.gpu_options.visible_device_list = '0'
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
            keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

    def get_fit_params(self):
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.logdir)
        es = keras.callbacks.EarlyStopping()
        checkpoint = ModelCheckpoint(f'{self.model_name}.best.hdf5', verbose=1, save_best_only=True, mode='max')

        fit_params = {
            'model__epochs': self.epochs,
            'model__batch_size': self.batch_size,
            'model__callbacks': [tensorboard_callback, es, checkpoint],
            'model__name': self.model_name,
        }

        return fit_params

    def _add_layers(self, input, output_dims, init_model=None, name=None):
        if init_model is None:  # new model to start with
            model = input
        else:
            model = init_model

        for i, out_dim in enumerate(output_dims):
            if out_dim == 1:
                activation = 'sigmoid'
            else:
                activation = self.activation

            model = Dense(out_dim, name=self.get_layer_name(i, "dense", name))(model)
            model = Activation(activation, name=self.get_layer_name(i, "activation", name))(model)

            if out_dim != 1:
                model = Dropout(rate=self.dropout_rate, name=self.get_layer_name(i, "dropout", name))(model)
        return model

    def get_optimizer(self):
        assert self.optimizer in ['adam', 'adadelta', 'adagrad',
                                  'adamax'], "Selected optimizer isn't supported, please use supported optimizer or add optimizer option in DeepBaseModel class"

        if self.optimizer == 'adam':
            optimzer = keras.optimizers.Adam(lr=self.initial_learning_rate)
        elif self.optimizer == 'adadelta':
            optimzer = keras.optimizers.Adadelta(lr=self.initial_learning_rate)
        elif self.optimizer == 'adagrad':
            optimzer = keras.optimizers.Adagrad(lr=self.initial_learning_rate)
        elif self.optimizer == 'adamax':
            optimzer = keras.optimizers.Adamax(lr=self.initial_learning_rate)
        return optimzer


    def build_nn(self):
        model = self.get_model()
        optimizer = self.get_optimizer()
        loss = self.loss()
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=['accuracy', 'mean_squared_error'])

        fit_params = self.get_fit_params()
        logging.info(model.summary())
        return model, fit_params

    def get_model(self):
        raise NotImplemented

    def get_layer_name(self, i, layer_type_name, name=None):
        return f'{layer_type_name}_{i + 1}{f"_{name}" if name is not None else ""}'