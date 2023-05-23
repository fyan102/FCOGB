import csv

import numpy as np
from sklearn.model_selection import train_test_split


def preprocess_csv(filename, features: list, types: list, target, target_type=int, feature_map={}):
    xx = []
    yy = []
    d = len(features)
    L = [None] * d
    U = [None] * d
    y_index = -1
    x_index = [-1] * d
    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile)
        first = True
        second = True
        for row in spamreader:
            if first:
                print(row)
                for i in range(len(row)):
                    if row[i] == target:
                        y_index = i
                    if row[i] in features:
                        x_index[features.index(row[i])] = i
                print(x_index, y_index)
                first = False
            else:
                try:
                    if second:
                        L = [types[j](row[x_index[j]]) if types[j] in [int, float] else feature_map[features[j]][
                            row[x_index[j]]] for j in range(len(x_index))]
                        U = [types[j](row[x_index[j]]) if types[j] in [int, float] else feature_map[features[j]][
                            row[x_index[j]]] for j in range(len(x_index))]
                        second = False
                    for i in range(len(x_index)):
                        value = types[i](row[x_index[i]]) if types[i] in [int, float] else feature_map[features[i]][
                            row[x_index[i]]]
                        if value < L[i]:
                            L[i] = value
                        elif value > U[i]:
                            U[i] = value
                    xx.append(
                        [types[j](row[x_index[j]]) if types[j] in [int, float] else feature_map[features[j]][
                            row[x_index[j]]] for j in range(len(x_index))])
                    yy.append(
                        target_type(row[y_index]) if target not in feature_map else feature_map[target][row[y_index]])
                except:
                    continue

            return xx, yy, L, U, d, len(yy)


def preprocess_pd(filename, features: list, types: list, target, target_type=int, feature_map={}, test_size=0.2,
                  random_seed=None):
    xx = []
    yy = []
    d = len(features)
    L = [None] * d
    U = [None] * d
    y_index = -1
    x_index = [-1] * d
    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile)
        first = True
        second = True
        for row in spamreader:
            if first:
                # print(row)
                for i in range(len(row)):
                    a = row[i]
                    if a == target:
                        y_index = i
                    if a in features:
                        x_index[features.index(a)] = i
                # print(x_index, y_index)
                first = False
            else:
                try:
                    xx.append(
                        [types[j](row[x_index[j]]) if types[j] in [int, float] else feature_map[features[j]][
                            row[x_index[j]]] for j in range(len(x_index))])
                    yy.append(
                        target_type(row[y_index]) if target not in feature_map else feature_map[target][row[y_index]])
                except Exception as e:
                    pass
                    # print('Error in preprocessing', e)
        mean = np.array(xx).mean(axis=0)
        std = np.array(xx).std(axis=0)
        # xx = [[(xx[i][j] - mean[j]) / std[j] for j in range(len(mean))] for i in range(len(xx))]
        train, test, train_target, test_target = train_test_split(xx, yy, random_state=random_seed, test_size=test_size)
        for i in range(len(train)):
            if i == 0:
                L = [x for x in train[0]]
                U = [x for x in train[0]]
            for j in range(len(train[i])):
                if L[j] > train[i][j]:
                    L[j] = train[i][j]
                elif U[j] < train[i][j]:
                    U[j] = train[i][j]
        n = len(train)
        return train, test, train_target, test_target, L, U, d, n
        # return xx, xx, yy, yy, L, U, d, len(xx)


def preprocess_datasets(load_dataset, feature_map={},
                        test_size=0.2,
                        random_seed=None):
    dataset = load_dataset()
    xx = dataset.data
    yy = dataset.target
    d = len(dataset.feature_names)
    L = [None] * d
    U = [None] * d
    if len(feature_map) != 0:
        for i in range(len(yy)):
            try:
                yy[i] = feature_map['target'][yy[i]]
            except Exception as e:
                print(e)
    mean = np.array(xx).mean(axis=0)
    std = np.array(xx).std(axis=0)
    # xx = [[(xx[i][j] - mean[j]) / std[j] for j in range(len(mean))] for i in range(len(xx))]
    train, test, train_target, test_target = train_test_split(xx, yy, random_state=random_seed,
                                                              test_size=test_size)
    for i in range(len(train)):
        if i == 0:
            L = [x for x in train[0]]
            U = [x for x in train[0]]
        for j in range(len(train[i])):
            if L[j] > train[i][j]:
                L[j] = train[i][j]
            elif U[j] < train[i][j]:
                U[j] = train[i][j]
    n = len(train)
    labels = dataset.feature_names
    return train, test, train_target, test_target, L, U, d, n, labels


def preprocess_gen(x, y, test_size=0.2, random_seed=None):
    d = len(x[0])
    L = [None] * d
    U = [None] * d
    mean = np.array(x).mean(axis=0)
    std = np.array(x).std(axis=0)
    # x = [[(x[i][j] - mean[j]) / std[j] for j in range(len(mean))] for i in range(len(x))]
    train, test, train_target, test_target = train_test_split(x, y, random_state=random_seed,
                                                              test_size=test_size)
    for i in range(len(train)):
        if i == 0:
            L = [xx for xx in train[0]]
            U = [xx for xx in train[0]]
        for j in range(len(train[i])):
            if L[j] > train[i][j]:
                L[j] = train[i][j]
            elif U[j] < train[i][j]:
                U[j] = train[i][j]
    n = len(train)
    return train, test, train_target, test_target, L, U, d, n
