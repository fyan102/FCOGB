from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from realkd.rules import loss_function, AdditiveRuleEnsemble

def cv(x, y, estimator, labels, loss='squared'):
    kf = KFold(n_splits=5)
    x = np.array(x)
    y = np.array(y)
    loss_func = loss_function(loss)
    sum_scores = 0
    origin_rules = AdditiveRuleEnsemble([rule for rule in estimator.rules_])
    for train_index, test_index in kf.split(x):
        train_x = pd.DataFrame(x[train_index], columns=labels)
        train_y = pd.Series(y[train_index])
        test_x = pd.DataFrame(x[test_index], columns=labels)
        test_y = pd.Series(y[test_index])
        if estimator.rules_ is None or len(estimator.rules_) == 0:
            estimator.fit(train_x, train_y)
        else:
            estimator.fit(train_x, train_y)
        # for ensemble in estimator.history:
        ensemble = estimator.rules_
        score = sum(loss_func(test_y, ensemble(test_x)))
        print("n:", len(ensemble), "score: ", score)
        estimator.rules_ = AdditiveRuleEnsemble([rule for rule in origin_rules])
        estimator.history.pop()
    return score

