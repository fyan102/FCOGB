from pandas import Series

from sklearn.metrics import roc_auc_score as skl_auc, \
                            accuracy_score as skl_accuracy,\
                            log_loss as skl_log_loss, \
                            r2_score as skl_r2


def r2(data, target):

    def metric(model):
        return skl_r2(target, model.predict(data))

    return metric


def accuracy(data, target):

    def metric(model):
        return skl_accuracy(target, model.predict(data))

    return metric


def roc_auc(data, target):

    def metric(model):
        if callable(model):
            return skl_auc(target, model(data))
        else:
            return skl_auc(target, model.predict_proba(data)[:, 1])

    return metric


def log_loss(data, target):

    def metric(model):
        return skl_log_loss(target, model.predict_proba(data)[:, 1])

    return metric


def ensemble_length_vs_perf(model, metric):
    """ Convenience method for evaluating ensemble size / performance trade-off

    :param model: ensemble model that allows to build prefixes by slicing
    :param metric: evaluation metric function that accepts as single argument a model
    :return: pandas Series of ensemble performance for all sequences (including the empty and full prefix)
    """
    res = Series()
    for k in range(len(model.members)+1):
        _model = model[:k]
        res.at[k] = metric(_model)
    return res
