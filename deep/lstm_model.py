from deep.attention import AttentionWithContext
from keras import Model, Input
from keras.layers import Dense, Dropout, Activation
from deep.deep_sequential_model import DeepSequentialModel


class LstmModel(DeepSequentialModel):
    def __init__(self, model_name = 'deep_model',
                 activation = 'sigmoid',
                 loss = 'binary_crossentropy',
                 optimizer = 'adam',
                 batch_size = 128,
                 dropout_rate = 0.4,
                 epochs = 50,
                 initial_learning_rate = 0.001,
                 vector_size = 300,
                 use_gpu = True,
                 use_attention = True,
                 layers_dim = [64,32],
                 lstm_layers = [64,32],
                 bidirectional_lstm = True):
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
            lstm_layers: list. list of lstm layers dimensions.
            bidirectional_lstm: boolean [True, False]. use bidirectional lstm instead of 1 directional lstm.
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
        self.lstm_layers = lstm_layers
        self.bidirectional_lstm = bidirectional_lstm

    def build_sequential_network(self):
        # input shape sequence length X embedding size
        codes_input = Input(shape = (self.sequence_length, self.vector_size))
        fc_input = self._add_layers(codes_input, self.layers_dim, name = f'input')
        lstm_output = self._add_lstm_layers(fc_input, self.lstm_layers, self.bidirectional_lstm, self.use_attention,
                                            name = f'lstm_input')
        if self.use_attention:
            lstm_output = AttentionWithContext()(lstm_output)
            lstm_output = Dropout(self.dropout_rate, name = f'att_dropout')(lstm_output)

        model = Dense(1, name = 'output')(lstm_output)
        model = Activation('sigmoid')(model)
        model = Model(inputs = codes_input, outputs = model, name = 'Final_output')
        return model
