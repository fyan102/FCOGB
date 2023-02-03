from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from realkd.rules import loss_function

def cv(x, y, estimator, labels, loss='squared'):
    kf = KFold(n_splits=5)
    x = np.array(x)
    y = np.array(y)
    loss_func = loss_function(loss)
    sum_scores = 0
    for train_index, test_index in kf.split(x):
        train_x = pd.DataFrame(x[train_index], columns=labels)
        train_y = pd.Series(y[train_index])
        test_x = pd.DataFrame(x[test_index], columns=labels)
        test_y = pd.Series(y[test_index])
        estimator.fit(train_x, train_y)
        for ensemble in estimator.history:
            score = sum(loss_func(test_y, ensemble(test_x))) / len(test_y)
            sum_scores += score
            print("n:", len(ensemble), "score: ", score)
    return sum_scores / 50

