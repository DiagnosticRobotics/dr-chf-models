from keras import Model, Input
from deep.deep_utils import DeepBaseModel


class DeepModel(DeepBaseModel):
    def __init__(self, model_name = 'deep_model',
                 activation = 'sigmoid',
                 loss = 'binary_crossentropy',
                 optimizer = 'adam',
                 batch_size = 128,
                 dropout_rate = 0.4,
                 epochs = 50,
                 initial_learning_rate = 0.001,
                 vector_size = 300,
                 use_gpu= True,
                 layers_dim = [64, 32]):
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
                         use_gpu)
        self.layers_dim = layers_dim
        self.config_gpu()

    def get_model(self):
        # all features to fc network
        output_dimensions = self.layers_dim + [1]
        features_input = Input(shape = (self.vector_size,))
        model = self._add_layers(features_input, output_dimensions)
        model = Model(inputs = features_input, outputs = model, name = 'Final_output')
        return model
