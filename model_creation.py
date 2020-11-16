from deep.cnn_model import CnnModel
from deep.deep_model import DeepModel
from deep.lstm_model import LstmModel
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def return_clf(model_type):
    """ return clf for a given parameters """
    # build model
    if  model_type == 'logistic_regression':
        model = LogisticRegression()

    elif model_type == 'XGBoost':
        model = XGBClassifier()

    elif model_type == "Deep":
        deep_model = DeepModel()
        model, fit_params = deep_model.build_nn()
        model.model_name = deep_model.model_name

    elif model_type == "DeepCNN":
        model = CnnModel()
        fit_params = model.get_fit_params()

    elif model_type == "DeepLstm":
        model = LstmModel()
        fit_params = model.get_fit_params()

    else:
        raise ValueError('unknown model type: {}'.format(model_type))

    if "Deep" not in model_type:
        model.model_name = model_type
        fit_params = {}

    return model, fit_params