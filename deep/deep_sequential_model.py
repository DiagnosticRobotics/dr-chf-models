import keras
from keras.layers import LSTM, Bidirectional
import multiprocessing as mp
import numpy as np
from deep.deep_utils import DeepBaseModel
from sklearn.base import TransformerMixin


class DeepSequentialModel(DeepBaseModel, TransformerMixin):
    '''
    The class is deep model relevant to sequence data - T (length of sequence) X vector_size (embedding size),
    uses multiple type of aggregations- cnn, attention and lstm
    '''

    SEQUENTIAL_MODEL_NAMES = ['DeepLstm', 'DeepCNN']

    def __init__(self,
                 model_name = 'deep_model',
                 activation = 'sigmoid',
                 loss = 'binary_crossentropy',
                 optimizer = 'adam',
                 batch_size = 128,
                 dropout_rate = 0.4,
                 epochs = 50,
                 initial_learning_rate = 0.001,
                 embedding_size = 100,
                 use_gpu = True,
                 use_attention = True,
                 layers_dim = [64, 32],
                 bucket_num = 1, time_granularity = 'month', last_years_number = 5):
        """

        Args:
            model_name: str. choose a name for the model
            activation: str. activation function for the deep layers. for example, 'relu', 'sigmoid'.
            loss: str. loss function.
            optimizer: str. deep model optimizer. choose from ['adam', 'adadelta', 'adagrad', 'adamax']
            batch_size: int. batch size.
            dropout_rate: float. dropout layer rate
            epochs: int. number of epochs.
            initial_learning_rate:  float. initial learning model for the deep model.
            embedding_size: int. embedding vector size.
            use_gpu: boolean [True, False]
            use_attention: boolean [True, False] to use attention layer on the sequence.
            layers_dim: list. list of fully connected layers dimensions.
            bucket_num: int. number of time units in one bucket. bucket_num=1 and time_granularity='month' represent
            a 1 month time data per element in the sequence.
            time_granularity: str.  'month', 'week' or 'day'
            last_years_number: int. number of years for the patient history. last 5 years with 1 month bucket will
            create a sequence of length 60.
        """
        super().__init__(model_name, activation, loss,
                         optimizer,
                         batch_size,
                         dropout_rate,
                         epochs,
                         initial_learning_rate,
                         embedding_size,
                         use_gpu)

        self.T = get_total_buckets_number(bucket_size = bucket_num,
                                          granularity = time_granularity,
                                          years_num = last_years_number)
        self.layers_dim = layers_dim
        self.use_attention = use_attention
        self.config_gpu()

    def get_model(self):
        return self.build_sequential_network()

    def fit(self, X, y = None, **fit_params):
        X_validation, Y_validation = fit_params['validation_data']
        sample_weights = fit_params.pop('sample_weight')
        training_generator = DataGenerator(x = X, y = y, model_config = self,
                                           sample_weights = sample_weights)
        validation_generator = DataGenerator(x = X_validation, y = Y_validation, model_config = self)
        fit_params['validation_data'] = validation_generator
        fit_params.pop('batch_size')

        # Train model on dataset
        self.model, _ = self.build_nn()
        self.model.fit_generator(generator = training_generator, **fit_params)

    def predict(self, X):
        data_generator = DataGenerator(x = X, model_config = self)
        return self.model.predict_generator(generator = data_generator, workers = mp.cpu_count())

    def load_weights(self, file):
        return self.model.load_weights(file)

    def build_sequential_network(self):
        raise NotImplemented

    def _add_lstm_layers(self, model, lstm_layers, bidirectional_lstm, return_sequences, name = 'lstm_hidden'):
        for i, layer_dim in enumerate(lstm_layers):
            if i == len(lstm_layers) - 1 and not return_sequences:  # last layer
                lstm_layer = LSTM(layer_dim, return_sequences = False, dropout = self.dropout_rate,
                                  name = name + f'_{i}')
            else:
                lstm_layer = LSTM(layer_dim, return_sequences = True, dropout = self.dropout_rate,
                                  name = name + f'_{i}')
            if bidirectional_lstm:
                model = Bidirectional(lstm_layer)(model)
            else:
                model = lstm_layer(model)
        return model


def get_total_buckets_number(bucket_size, granularity, years_num):
    if granularity == 'month':
        return int(years_num * 12 / bucket_size)
    elif granularity == 'week':
        return int(years_num * 52 / bucket_size)
    elif granularity == "day":
        return int(years_num * 365 / bucket_size)
    else:
        raise Exception("Unimplemented granularity type, supporting only 'month','week' or 'day'")


class DataGenerator(keras.utils.Sequence):
    'Generates data in batches'

    def __init__(self, x, model_config, y = None, sample_weights = None):
        super().__init__()
        self.embedding_size = model_config.embedding_size
        self.input_params_list = []
        self.batch_size = model_config.batch_size
        self.T = model_config.T
        self.ids = x.index.values
        self.y = y
        self.weights = sample_weights
        self.shuffle = False
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of ids
        batch_list_ids = [self.ids[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(batch_list_ids)

        if self.weights is not None:
            return X, y, self.weights[indexes]

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids):
        'Generates data containing batch_size samples'  # X : (n_samples, T,  vec_size)
        # Initialization
        X = np.empty((len(list_ids), self.T, self.embedding_size))
        y = np.empty(len(list_ids), dtype = int)
        # Generate data
        for i, id in enumerate(list_ids):
            # Store sample
            X[i,] = np.load(f'{id}.npy')
            # Store class
            if self.y is not None:
                y[i] = self.y[id]
        return X, y
