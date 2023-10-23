import os
from datetime import datetime
from math import log, exp

import numpy as np
import pandas as pd
from realkd.boosting import GradientBoostingObjectiveMWG, GradientBoostingObjectiveGPE, OrthogonalBoostingObjective, \
    OrthogonalBoostingObjectiveSlow, FullyCorrective, KeepWeight, LineSearch, GeneralRuleBoostingEstimator
from realkd.rules import loss_function, GradientBoostingObjective, Rule, AdditiveRuleEnsemble
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3

from evaluation.cross_validation import cv
from evaluation.data_info import get_splits
from evaluation.data_preprocess import preprocess_datasets, preprocess_pd, preprocess_gen

objs = {'xgb': GradientBoostingObjective, 'mwg': GradientBoostingObjectiveMWG, 'gpe': GradientBoostingObjectiveGPE,
        'orth': OrthogonalBoostingObjective, 'orth_slow': OrthogonalBoostingObjectiveSlow,
        }
weight_upds = {'boosting': LineSearch, 'fc': FullyCorrective, 'keep': KeepWeight}
folder = "../experiment_output_"


def evaluate_dataset(dataset_name, path, labels, feature_types, target, target_type=int, obj='orth',
                     weight_update='fc', weight_update_method='Line', feature_map={}, loss='squared',
                     search='exhaustive',
                     repeat=5, max_rule_num=5, regs=(0, 0.1, 0.2, 0.5, 0.7, 1, 2, 4, 8, 16), col=10):
    print('==========', dataset_name, '===========')
    print(obj, weight_update, weight_update_method)
    print('---------------------------------------')
    seeds = get_splits()[dataset_name]
    fc_train_risk_all = []
    fc_test_risk_all = []
    fc_coverages_all = []
    for m in range(repeat):
        selected_regs = []
        fc_risk = []
        fc_train_risk = []
        fc_test_risk = []
        fc_coverages = []
        loss_func = loss_function(loss)
        fc_ensembles = []
        obj_function = objs[obj]
        weight_update_func = weight_upds[weight_update]() if weight_update != 'fc' \
            else weight_upds[weight_update](solver=weight_update_method)

        if not os.path.exists(folder + search):
            os.makedirs(folder + search)
        if not os.path.exists(folder + search + "/" + dataset_name):
            os.makedirs(folder + search + "/" + dataset_name)
        output = open(
            folder + search + "/" + dataset_name + '/' + dataset_name + '_' + obj + '_' + weight_update + '_' +
            weight_update_method + '_realkd_col_' + str(col) + '_' + 'rep' + str(m) + ".txt", "w")
        train, test, train_target, test_target, _, _, _, n = preprocess_pd(path,
                                                                           labels,
                                                                           feature_types,
                                                                           target, target_type=target_type,
                                                                           feature_map=feature_map,
                                                                           random_seed=seeds[m])
        train_df = pd.DataFrame(train, columns=labels)
        test_df = pd.DataFrame(test, columns=labels)
        ys = np.array(train_target + test_target)
        a = ys.mean()
        b = ys.std()
        print(a, b)
        if loss == 'squared':
            # train_sr = pd.Series((train_target - a) / b)
            # test_sr = pd.Series((test_target - a) / b)
            train_sr = pd.Series(train_target)
            test_sr = pd.Series(test_target)
        else:
            train_sr = pd.Series(train_target)
            test_sr = pd.Series(test_target)
        # default_rule = AdditiveRuleEnsemble([Rule(y=sum(train_sr) / len(train_sr))])
        # print(default_rule)
        fc_estimator = GeneralRuleBoostingEstimator(num_rules=1,
                                                    max_col_attr=col, search=search,
                                                    objective_function=obj_function,
                                                    weight_update_method=weight_update_func,
                                                    # fit_intercept=True, normalize=True,
                                                    # init_ensemble=default_rule,
                                                    loss=loss)
        for i in range(1, 1 + max_rule_num):
            fc_estimator.num_rules = i
            scores = {}
            if len(regs) == 1:
                reg = regs[0]
            else:
                origin_rules = AdditiveRuleEnsemble([rule for rule in fc_estimator.rules_])
                for r in regs:
                    print('--------', r, '--------')
                    fc_estimator.set_reg(r)
                    scores[r] = cv(train, train_target, fc_estimator, labels, loss=loss)
                    fc_estimator.rules_ = AdditiveRuleEnsemble([rule for rule in origin_rules])
                    # fc_estimator.history.pop()
                print('fc scores:', scores)
                # find best lambda
                reg = list(scores.keys())[0]
                for r in scores:
                    if scores[r] < scores[reg]:
                        reg = r
            selected_regs.append(reg)
            fc_estimator.set_reg(reg)
            # try:
            start_time = datetime.now()
            fc_estimator.fit(train_df, train_sr)
            end_time = datetime.now()
            print('runnning time:', end_time - start_time)
            output.write('Running time:' + str(end_time - start_time) + '\n')
            # output.write('Each rule: ' + str(fc_estimator.time))
            # print(fc_estimator.rules_)
            fc_ensemble = fc_estimator.rules_
            # if loss == 'squared':
            #     train_sr = train_sr * b + a
            #     test_sr = test_sr * b + a
            # for fc_ensemble in fc_estimator.history:
            # if loss == 'squared':
            #     risk = sum(loss_func(train_sr, fc_ensemble(train_df) * b + a)) / n + reg * sum(
            #         [rule.y * rule.y for rule in fc_ensemble.members]) / 2 / n
            #     test_risk = sum(loss_func(test_sr, fc_ensemble(test_df) * b + a)) / len(test_sr)
            #     train_risk = sum(loss_func(train_sr, fc_ensemble(train_df) * b + a)) / n
            # else:
            risk = sum(loss_func(train_sr, fc_ensemble(train_df))) / n + reg * sum(
                [rule.y * rule.y for rule in fc_ensemble.members]) / 2 / n
            test_risk = sum(loss_func(test_sr, fc_ensemble(test_df))) / len(test_sr)
            train_risk = sum(loss_func(train_sr, fc_ensemble(train_df))) / n
            fc_test_risk.append(test_risk)
            fc_train_risk.append(train_risk)
            fc_risk.append(risk)
            fc_ensembles.append(str(fc_ensemble))
            coverage = sum(fc_ensemble[-1].q(train_df))
            fc_coverages.append(coverage)
            print(fc_ensemble)
            print('coverage', coverage)
            print('risk', risk)
            print('train_risk', train_risk, 'test_risk', test_risk)
            fc_train_risk_all.append(sum(fc_train_risk) / len(fc_train_risk))
            fc_test_risk_all.append(sum(fc_test_risk) / len(fc_test_risk))
            fc_coverages_all.append(fc_coverages)
            # except Exception as e:
            #     print('Error2: ', e)
            try:
                # for i in range(max_rule_num):
                output.write('\n=======iteration ' + str(i) + '========\n')

                output.write('\nfc risk: ' + str(risk) + '\n')
                output.write('fc train risk: ' + str(train_risk) + '\n')
                output.write('fc test risk: ' + str(test_risk) + '\n')
                output.write('coverage: ' + str(coverage) + '\n')
                output.write(str(fc_ensemble))
            except Exception as e:
                print('Error6: ', e)
            output.write('reg: ' + str(reg))
            s = str(fc_ensemble)
            cnt = s.count('=') + s.count('if ')
            if cnt >= 100:
                break

        output.close()
    return fc_train_risk_all, fc_test_risk_all, fc_coverages_all


def evaluate_loaded_data(dataset_name, load_method, obj='xgb',
                         weight_update='fc', weight_update_method='Line', feature_map={}, loss='squared',
                         search='exhaustive',
                         repeat=5, max_rule_num=5, regs=(0, 0.1, 0.2, 0.5, 0.7, 1, 2, 4, 8, 16), col=10):
    print('==========', dataset_name, '===========')
    print(obj, weight_update, weight_update_method)
    print('---------------------------------------')
    seeds = get_splits()[dataset_name]
    fc_train_risk_all = []
    fc_test_risk_all = []
    fc_coverages_all = []
    for m in range(repeat):
        selected_regs = []
        fc_risk = []
        fc_train_risk = []
        fc_test_risk = []
        fc_coverages = []
        loss_func = loss_function(loss)
        fc_ensembles = []
        obj_function = objs[obj]
        weight_update_func = weight_upds[weight_update]() if weight_update != 'fc' \
            else weight_upds[weight_update](solver=weight_update_method)
        fc_estimator = GeneralRuleBoostingEstimator(num_rules=max_rule_num,
                                                    max_col_attr=col, search=search,
                                                    objective_function=obj_function,
                                                    weight_update_method=weight_update_func,
                                                    # fit_intercept=True, normalize=True,
                                                    loss=loss)
        if not os.path.exists(folder + search):
            os.makedirs(folder + search)
        if not os.path.exists(folder + search + "/" + dataset_name):
            os.makedirs(folder + search + "/" + dataset_name)
        output = open(
            folder + search + "/" + dataset_name + '/' + dataset_name + '_' + obj + '_' + weight_update + '_' +
            weight_update_method + '_realkd_col_' + str(col) + '_' + 'rep' + str(m) + ".txt", "w")
        train, test, train_target, test_target, _, _, _, n, labels = preprocess_datasets(load_method,
                                                                                         feature_map=feature_map,
                                                                                         random_seed=seeds[m])
        train_df = pd.DataFrame(train, columns=labels)
        test_df = pd.DataFrame(test, columns=labels)
        ys = np.concatenate((train_target, test_target))
        # a = ys.mean()
        # b = ys.std()
        if loss == 'squared':
            train_sr = pd.Series(train_target)
            test_sr = pd.Series(test_target)
        else:
            train_sr = pd.Series(train_target)
            test_sr = pd.Series(test_target)
        scores = {}
        for i in range(1, 1 + max_rule_num):
            fc_estimator.num_rules = i
            scores = {}
            if len(regs) == 1:
                reg = regs[0]
            else:
                if i == 6:
                    regs = regs[1:]
                origin_rules = AdditiveRuleEnsemble([rule for rule in fc_estimator.rules_])
                for r in regs:
                    print('--------', r, '--------')
                    fc_estimator.set_reg(r)
                    scores[r] = cv(train, train_target, fc_estimator, labels, loss=loss)
                    fc_estimator.rules_ = AdditiveRuleEnsemble([rule for rule in origin_rules])
                    # fc_estimator.history.pop()
                print('fc scores:', scores)
                # find best lambda
                reg = list(scores.keys())[0]
                for r in scores:
                    if scores[r] < scores[reg]:
                        reg = r
            selected_regs.append(reg)
            fc_estimator.set_reg(reg)
            start_time = datetime.now()
            fc_estimator.fit(train_df, train_sr)
            end_time = datetime.now()
            print('runnning time:', end_time - start_time)
            output.write('Running time:' + str(end_time - start_time) + '\n')
            output.write('Each rule: ' + str(fc_estimator.time))
            # print(fc_estimator.rules_)
            fc_ensemble = fc_estimator.rules_
            # if loss == 'squared':
            #     train_sr = train_sr
            #     test_sr = test_sr
            # for fc_ensemble in fc_estimator.history:
            #     if loss == 'squared':
            #         risk = sum(loss_func(train_sr, fc_ensemble(train_df) * b + a)) / n + reg * sum(
            #             [rule.y * rule.y for rule in fc_ensemble.members]) / 2 / n
            #         test_risk = sum(loss_func(test_sr, fc_ensemble(test_df) * b + a)) / len(test_sr)
            #         train_risk = sum(loss_func(train_sr, fc_ensemble(train_df) * b + a)) / n
            #     else:
            risk = sum(loss_func(train_sr, fc_ensemble(train_df))) / n + reg * sum(
                [rule.y * rule.y for rule in fc_ensemble.members]) / 2 / n
            test_risk = sum(loss_func(test_sr, fc_ensemble(test_df))) / len(test_sr)
            train_risk = sum(loss_func(train_sr, fc_ensemble(train_df))) / n
            fc_test_risk.append(test_risk)
            fc_train_risk.append(train_risk)
            fc_risk.append(risk)
            fc_ensembles.append(str(fc_ensemble))
            coverage = sum(fc_ensemble[-1].q(train_df))
            fc_coverages.append(coverage)
            print(fc_ensemble)
            print('coverage', coverage)
            print('risk', risk)
            print('train_risk', train_risk, 'test_risk', test_risk)
            fc_train_risk_all.append(sum(fc_train_risk) / len(fc_train_risk))
            fc_test_risk_all.append(sum(fc_test_risk) / len(fc_test_risk))
            fc_coverages_all.append(fc_coverages)

            try:
                # for i in range(max_rule_num):
                output.write('\n=======iteration ' + str(i) + '========\n')
                # if i < len(fc_risk):
                output.write('\nfc risk: ' + str(fc_risk[-1]) + '\n')
                output.write('fc train risk: ' + str(fc_train_risk[-1]) + '\n')
                output.write('fc test risk: ' + str(fc_test_risk[-1]) + '\n')
                output.write('coverage: ' + str(fc_coverages[-1]) + '\n')
                output.write(fc_ensembles[-1])
            except Exception as e:
                print('Error6: ', e)

            s = str(fc_ensemble)
            cnt = s.count('=') + s.count('if ')
            if cnt > 50:
                break
        output.write(str(selected_regs))
        output.close()
    return fc_train_risk_all, fc_test_risk_all, fc_coverages_all


def gen_friedman(func_name, n, noise, random_seed, d=4):
    func_map = {'make_friedman1': make_friedman1, 'make_friedman2': make_friedman2, 'make_friedman3': make_friedman3, }
    if func_name == 'make_friedman1':
        x, y = func_map[func_name](n_samples=n, n_features=d, noise=noise, random_state=random_seed)
    else:
        x, y = func_map[func_name](n_samples=n, noise=noise, random_state=random_seed)
    labels = ['x' + str(i) for i in range(1, d + 1)]
    return x, y, labels


def evaluate_friedman(dataset_name, number, noise, d=4, test_size=0.2, obj='xgb',
                      weight_update='fc', weight_update_method='Line', feature_map={}, loss='squared',
                      search='exhaustive',
                      repeat=5, max_rule_num=5, regs=(0, 0.1, 0.2, 0.5, 0.7, 1, 2, 4, 8, 16), col=10):
    print('==========', dataset_name, '===========')
    print(obj, weight_update, weight_update_method)
    print('---------------------------------------')
    seeds = get_splits()[dataset_name]
    fc_train_risk_all = []
    fc_test_risk_all = []
    fc_coverages_all = []
    for m in range(repeat):
        selected_regs = []
        fc_risk = []
        fc_train_risk = []
        fc_test_risk = []
        fc_coverages = []
        loss_func = loss_function(loss)
        fc_ensembles = []
        obj_function = objs[obj]
        weight_update_func = weight_upds[weight_update]() if weight_update != 'fc' \
            else weight_upds[weight_update](solver=weight_update_method)
        fc_estimator = GeneralRuleBoostingEstimator(num_rules=max_rule_num,
                                                    max_col_attr=col, search=search,
                                                    objective_function=obj_function,
                                                    weight_update_method=weight_update_func,
                                                    # fit_intercept=True, normalize=True,
                                                    loss=loss)
        x, y, labels = gen_friedman(dataset_name, number, noise, 1000, d=d)
        if not os.path.exists(folder + search):
            os.makedirs(folder + search)
        if not os.path.exists(folder + search + "/" + dataset_name):
            os.makedirs(folder + search + "/" + dataset_name)
        output = open(
            folder + search + "/" + dataset_name + '/' + dataset_name + '_' + obj + '_' + weight_update + '_' +
            weight_update_method + '_realkd_col_' + str(col) + '_' + 'rep' + str(m) + ".txt", "w")
        train, test, train_target, test_target, _, _, _, n = preprocess_gen(x, y, test_size=test_size,
                                                                            random_seed=seeds[m])
        print(train[0], train_target[0])
        train_df = pd.DataFrame(train, columns=labels)
        test_df = pd.DataFrame(test, columns=labels)
        ys = np.concatenate((train_target, test_target))
        a = ys.mean()
        b = ys.std()
        if loss == 'squared':
            train_sr = pd.Series(train_target)
            test_sr = pd.Series(test_target)
        else:
            train_sr = pd.Series(train_target)
            test_sr = pd.Series(test_target)
        scores = {}
        for i in range(1, 1 + max_rule_num):
            fc_estimator.num_rules = i
            scores = {}
            if len(regs) == 1:
                reg = regs[0]
            else:
                origin_rules = AdditiveRuleEnsemble([rule for rule in fc_estimator.rules_])
                for r in regs:
                    print('--------', r, '--------')
                    fc_estimator.set_reg(r)
                    scores[r] = cv(train, train_target, fc_estimator, labels, loss=loss)
                    fc_estimator.rules_ = AdditiveRuleEnsemble([rule for rule in origin_rules])
                    # fc_estimator.history.pop()
                print('fc scores:', scores)
                # find best lambda
                reg = list(scores.keys())[0]
                for r in scores:
                    if scores[r] < scores[reg]:
                        reg = r
            selected_regs.append(reg)
            fc_estimator.set_reg(reg)
            start_time = datetime.now()
            fc_estimator.fit(train_df, train_sr)
            end_time = datetime.now()
            print('runnning time:', end_time - start_time)
            output.write('Running time:' + str(end_time - start_time) + '\n')
            # output.write('Each rule: ' + str(fc_estimator.time))
            # print(fc_estimator.rules_)
            fc_ensemble = fc_estimator.rules_
            risk = sum(loss_func(train_sr, fc_ensemble(train_df))) / n + reg * sum(
                [rule.y * rule.y for rule in fc_ensemble.members]) / 2 / n
            test_risk = sum(loss_func(test_sr, fc_ensemble(test_df))) / len(test_sr)
            train_risk = sum(loss_func(train_sr, fc_ensemble(train_df))) / n
            fc_test_risk.append(test_risk)
            fc_train_risk.append(train_risk)
            fc_risk.append(risk)
            fc_ensembles.append(str(fc_ensemble))
            coverage = sum(fc_ensemble[-1].q(train_df))
            fc_coverages.append(coverage)
            print(fc_ensemble)
            print('coverage', coverage)
            print('risk', risk)
            print('train_risk', train_risk, 'test_risk', test_risk)
            fc_train_risk_all.append(sum(fc_train_risk) / len(fc_train_risk))
            fc_test_risk_all.append(sum(fc_test_risk) / len(fc_test_risk))
            fc_coverages_all.append(fc_coverages)
            # except Exception as e:
            #     print('Error2: ', e)
            try:
                # for i in range(max_rule_num):
                output.write('\n=======iteration ' + str(i) + '========\n')

                output.write('\nfc risk: ' + str(fc_risk[-1]) + '\n')
                output.write('fc train risk: ' + str(fc_train_risk[-1]) + '\n')
                output.write('fc test risk: ' + str(fc_test_risk[-1]) + '\n')
                output.write('coverage: ' + str(fc_coverages[-1]) + '\n')
                output.write(str(fc_ensemble))
                output.write(str(train[0]) + ' ' + str(train_target[0]))
            except Exception as e:
                print('Error6: ', e)
            s = str(fc_ensemble)
            cnt = s.count('=') + s.count('if ')
            if cnt > 50:
                break
        output.write(str(selected_regs))
        output.close()
    return fc_train_risk_all, fc_test_risk_all, fc_coverages_all


def evaluate_poisson(dataset_name, path, labels, feature_types, target, target_type=int, obj='xgb',
                     weight_update='fc', weight_update_method='Line', feature_map={}, loss='squared',
                     search='exhaustive',
                     repeat=5, max_rule_num=5, regs=(0, 0.1, 0.2, 0.5, 0.7, 1, 2, 4, 8, 16), col=10):
    print('==========', dataset_name, '===========')
    print(obj, weight_update, weight_update_method)
    print('---------------------------------------')
    seeds = get_splits()[dataset_name]
    fc_train_risk_all = []
    fc_test_risk_all = []
    fc_coverages_all = []
    for m in range(repeat):
        selected_regs = []
        fc_risk = []
        fc_train_risk = []
        fc_test_risk = []
        fc_coverages = []
        loss_func = loss_function(loss)
        fc_ensembles = []
        obj_function = objs[obj]
        weight_update_func = weight_upds[weight_update]() if weight_update != 'fc' \
            else weight_upds[weight_update](solver=weight_update_method)
        fc_estimator = GeneralRuleBoostingEstimator(num_rules=max_rule_num,
                                                    max_col_attr=col, search=search,
                                                    objective_function=obj_function,
                                                    weight_update_method=weight_update_func,
                                                    loss=loss)
        if not os.path.exists(folder + search):
            os.makedirs(folder + search)
        if not os.path.exists(folder + search + "/" + dataset_name):
            os.makedirs(folder + search + "/" + dataset_name)
        output = open(
            folder + search + "/" + dataset_name + '/' + dataset_name + '_' + obj + '_' + weight_update + '_' +
            weight_update_method + '_realkd_col_' + str(col) + '_' + 'rep' + str(m) + ".txt", "w")
        train, test, train_target, test_target, _, _, _, n = preprocess_pd(path,
                                                                           labels,
                                                                           feature_types,
                                                                           target, target_type=target_type,
                                                                           feature_map=feature_map,
                                                                           random_seed=seeds[m])
        ys = train_target + test_target
        logys = [log(y) if y != 0 else 0 for y in ys]
        max_y = exp(sum(logys) / len(train_target + test_target))
        print("average log exp of y:", max_y)
        train_target = [x / max_y for x in train_target]
        test_target = [x / max_y for x in test_target]
        train_df = pd.DataFrame(train, columns=labels)
        test_df = pd.DataFrame(test, columns=labels)
        train_sr = pd.Series(train_target)
        test_sr = pd.Series(test_target)
        for i in range(1, 1 + max_rule_num):
            fc_estimator.num_rules = i
            scores = {}
            if len(regs) == 1:
                reg = regs[0]
            else:
                origin_rules = AdditiveRuleEnsemble([rule for rule in fc_estimator.rules_])
                for r in regs:
                    print('--------', r, '--------')
                    fc_estimator.set_reg(r)
                    scores[r] = cv(train, train_target, fc_estimator, labels, loss=loss)
                    fc_estimator.rules_ = AdditiveRuleEnsemble([rule for rule in origin_rules])
                    # fc_estimator.history.pop()
                print('fc scores:', scores)
                # find best lambda
                reg = list(scores.keys())[0]
                for r in scores:
                    if scores[r] < scores[reg]:
                        reg = r
            selected_regs.append(reg)
            fc_estimator.set_reg(reg)
            # try:
            start_time = datetime.now()
            fc_estimator.fit(train_df, train_sr)
            end_time = datetime.now()
            print('runnning time:', end_time - start_time)
            output.write('Running time:' + str(end_time - start_time) + '\n')
            output.write('Each rule: ' + str(fc_estimator.time))
            # print(fc_estimator.rules_)
            fc_ensemble = fc_estimator.rules_
            # for fc_ensemble in fc_estimator.history:
            risk = sum(loss_func(train_sr, fc_ensemble(train_df))) / n + reg * sum(
                [rule.y * rule.y for rule in fc_ensemble.members]) / 2 / n
            train_target1 = [x * max_y for x in train_target]
            test_target1 = [x * max_y for x in test_target]
            train_sr1 = pd.Series(train_target1)
            test_sr1 = pd.Series(test_target1)
            test_risk = sum(loss_func(test_sr, fc_ensemble(test_df))) / len(test_sr)
            train_risk = sum(loss_func(train_sr, fc_ensemble(train_df))) / n
            test_risk1 = sum(loss_func(test_sr1, fc_ensemble(test_df) + log(max_y))) / len(test_sr)
            train_risk1 = sum(loss_func(train_sr1, fc_ensemble(train_df) + log(max_y))) / n
            risk1 = sum(loss_func(train_sr1, fc_ensemble(train_df) + log(max_y))) / n + reg * sum(
                [rule.y * rule.y for rule in fc_ensemble.members]) / 2 / n
            fc_test_risk.append(test_risk1)
            fc_train_risk.append(train_risk1)
            fc_risk.append(risk1)
            fc_ensembles.append(str(fc_ensemble))
            coverage = sum(fc_ensemble[-1].q(train_df))
            fc_coverages.append(coverage)
            print(fc_ensemble)
            print('risk', risk)
            print('train_risk', train_risk, 'test_risk', test_risk)
            print('train_risk1', train_risk1, 'test_risk1', test_risk1)
            print('coverage', coverage)
            fc_train_risk_all.append(sum(fc_train_risk) / len(fc_train_risk))
            fc_test_risk_all.append(sum(fc_test_risk) / len(fc_test_risk))
            fc_coverages_all.append(fc_coverages)
            # except Exception as e:
            #     print('Error2: ', e)
            # for i in range(max_rule_num):
            output.write('\n=======iteration ' + str(i) + '========\n')
            # if i < len(fc_risk):
            output.write('\nfc risk: ' + str(fc_risk[-1]) + '\n')
            output.write('fc train risk: ' + str(fc_train_risk[-1]) + '\n')
            output.write('fc test risk: ' + str(fc_test_risk[-1]) + '\n')
            output.write('coverage: ' + str(fc_coverages[-1]) + '\n')
            output.write(fc_ensembles[-1])
            output.write('background rule: ' + str(log(max_y)) + ' if True')

            s = str(fc_ensemble)
            cnt = s.count('=') + s.count('if ')
            if cnt > 50:
                break
        output.write(str(selected_regs))
        output.close()
    return fc_train_risk_all, fc_test_risk_all, fc_coverages_all
