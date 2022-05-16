import numpy as np

def mse(actual, predicted):
    """ Mean Squared Error """
    return np.mean(np.square(np.subtract(actual, predicted)))

def rmse(actual, predicted):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))

def nrmse(actual, predicted):
    """ Normalized Root Mean Squared Error """
    return rmse(actual, predicted) / (actual.max() - actual.min())

def mae(actual, predicted):
    """ Mean Absolute Error """
    return np.mean(np.abs(np.subtract(actual, predicted)))

def wape(actual, predicted):
    """ Weighted Absolute Percentage Error """
    return mae(actual, predicted)/np.mean(actual)


METRICS = {
    'mse': mse,
    'rmse': rmse,
    'nrmse': nrmse,
    'mae': mae,
    'wape': wape
}

def evaluate_all(actual, predicted):
    return evaluate(actual, predicted, metrics=set(METRICS.keys()))

def evaluate(actual, predicted, metrics=('mae', 'mse', 'rmse', 'mape')):
    results = {}
    for name in metrics:
        results[name] = METRICS[name](actual, predicted)
    return results