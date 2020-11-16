from keras import Input, regularizers, Model
from keras.layers import Dropout, Concatenate, Reshape, Conv2D, MaxPool2D, Flatten, Dense, Activation
from deep.deep_sequential_model import DeepSequentialModel
from deep.attention import AttentionWithContext


class CnnModel(DeepSequentialModel):
    def __init__(self, model_name = 'deep_model',
                 activation = 'relu',
                 loss = 'binary_crossentropy',
                 optimizer = 'adam',
                 batch_size = 128,
                 dropout_rate = 0.4,
                 epochs = 50,
                 initial_learning_rate = 0.001,
                 vector_size = 300,
                 use_gpu = True,
                 use_attention = True,
                 layers_dim = [64, 32],
                 num_filters = 100,
                 kernel_size = [3, 6],
                 kernel_initializer = 'normal'):
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
            use_attention: boolean [True, False] to use attention layer on the sequence.
            layers_dim: list. list of fully connected layers dimensions.
            num_filters: int, the dimensionality of the output space (i.e. the number of output filters in the convolution).
            kernel_size: list of 2 integers, specifying the height and width of the 2D convolution window.
            kernel_initializer: str. Initializer for the kernel weights matrix.
        '''
        super().__init__(model_name,
                         activation,
                         loss,
                         optimizer,
                         batch_size,
                         dropout_rate,
                         epochs,
                         initial_learning_rate,
                         vector_size,
                         use_gpu,
                         use_attention, layers_dim)
        self.num_filters = num_filters
        self.kernels_size = kernel_size
        self.kernel_initializer = kernel_initializer

    def build_sequential_network(self):
        # input shape sequence length X embedding size
        codes_input = Input(shape = (self.sequence_length, self.vector_size))
        reshape = Reshape((self.sequence_length, self.vector_size, 1))(codes_input)
        # CNN layers
        layers = []
        for i, kernel_size in enumerate(self.kernels_size):
            model = Conv2D(self.num_filters, kernel_size = (kernel_size, self.vector_size),
                           activation = self.activation,
                           kernel_regularizer = regularizers.l2(), name = f'conv_{i}')(reshape)
            layers.append(MaxPool2D(pool_size = (self.sequence_length - kernel_size + 1, 1), strides = (1, 1), padding = 'valid',
                                    name = f'maxpooling_{i}')(model))

        # concat and flatten
        concatenated_layer = Concatenate(axis = 1, name = "concat_cnn")(layers)
        flatten = Flatten(name = "flatten")(concatenated_layer)
        if self.use_attention:
            flatten = AttentionWithContext()(flatten)
            flatten = Dropout(self.dropout_rate, name = f'att_dropout')(flatten)

        model = self._add_layers(flatten, self.layers_dim)
        dropout = Dropout(self.dropout_rate)(model)

        model = Dense(1, name = 'output')(dropout)
        model = Activation('sigmoid')(model)
        model = Model(inputs = codes_input, outputs = model, name = 'Final_output')
        return model
