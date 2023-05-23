import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split


def gen_friedman(func_name, n, noise, random_seed, d=4):
    func_map = {'make_friedman1': make_friedman1, 'make_friedman2': make_friedman2, 'make_friedman3': make_friedman3, }
    if func_name == 'make_friedman1':
        x, y = func_map[func_name](n_samples=n, n_features=d, noise=noise, random_state=random_seed)
    else:
        x, y = func_map[func_name](n_samples=n, noise=noise, random_state=random_seed)
    labels = ['x' + str(i) for i in range(1, d + 1)]
    dataset = pd.DataFrame(np.hstack([x, np.array([y]).T]), columns=labels + ['y'])
    dataset.to_csv('../datasets/' + func_name + '/' + func_name + '.csv', index=False)


def load_datasets(dataset_name, load_dataset, feature_map={}):
    dataset = load_dataset()
    xx = dataset.data
    yy = dataset.target
    if len(feature_map) != 0:
        for i in range(len(yy)):
            try:
                yy[i] = feature_map['target'][yy[i]]
            except Exception as e:
                print(e)

    labels = dataset.feature_names
    print(type(labels), labels)
    dataset_pd = pd.DataFrame(np.hstack([xx, np.array([yy]).T]), columns=np.append(np.array(labels), 'y'))
    dataset_pd.to_csv('../datasets/' + dataset_name + '/' + dataset_name + '.csv', index=False)


if __name__ == '__main__':
    gen_friedman('make_friedman1', 2000, 0.1, 1000, d=10)
    gen_friedman('make_friedman2', 5000, 0.1, 1000)
    gen_friedman('make_friedman3', 10000, 0.1, 1000)
    load_datasets('load_wine', load_wine,
                  feature_map={'target': {0: -1, 1: 1, 2: -1}},
                  )
    load_datasets('iris', load_iris, feature_map={'target': {0: -1, 1: 1, 2: -1}})
    load_datasets('load_diabetes', load_diabetes)
    load_datasets('breast_cancer', load_breast_cancer, feature_map={'target': {0: -1, 1: 1}}, )
