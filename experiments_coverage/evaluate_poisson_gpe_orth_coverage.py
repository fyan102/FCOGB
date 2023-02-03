import warnings
from math import log, exp

import numpy as np
import pandas as pd
from pandas import qcut
from realkd.boosting import GeneralRuleBoostingEstimator, FullyCorrective, GradientBoostingObjectiveMWG, LineSearch, \
    GradientBoostingObjectiveGPE, KeepWeight, OrthogonalBoostingObjective
from realkd.rules import loss_function, XGBRuleEstimator, GradientBoostingObjective
from sklearn.model_selection import KFold

from evaluation.data_info import get_splits
from evaluation.data_preprocess import preprocess_pd


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


objs = {'xgb': GradientBoostingObjective, 'mwg': GradientBoostingObjectiveMWG, 'gpe': GradientBoostingObjectiveGPE,
        'orth': OrthogonalBoostingObjective}
weight_upds = {'boosting': LineSearch, 'fc': FullyCorrective, 'keep': KeepWeight}


def evaluate(dataset_name, path, labels, feature_types, target, target_type=int, obj='xgb',
             weight_update='fc', weight_update_method='GD', feature_map={}, loss='squared',
             repeat=5, max_rule_num=5, regs=(0, 0.1, 0.2, 0.5, 0.7, 1, 2, 4, 8, 16), col=10):
    print('==========', dataset_name, '===========')
    print(obj, weight_update, weight_update_method)
    print('---------------------------------------')
    seeds = get_splits()[dataset_name]
    fc_train_risk_all = []
    fc_test_risk_all = []
    fc_coverages_all = []
    orth_coverages_all = []
    for m in range(repeat):
        selected_regs = []
        fc_risk = []
        fc_train_risk = []
        fc_test_risk = []
        fc_coverages = []
        orth_coverages = []
        orth_fc_risk = []
        orth_fc_train_risk = []
        orth_fc_test_risk = []
        orth_fc_ensembles = []
        loss_func = loss_function(loss)
        fc_ensembles = []
        obj_function = objs[obj]
        weight_update_func = weight_upds[weight_update]() if weight_update != 'fc' \
            else weight_upds[weight_update](solver=weight_update_method)
        fc_estimator = GeneralRuleBoostingEstimator(num_rules=max_rule_num,
                                                    max_col_attr=col, search='exhaustive',
                                                    objective_function=obj_function,
                                                    weight_update_method=weight_update_func,
                                                    loss=loss)

        # output = open(
        #     "../output20221223realkd_cv_eval_no_reg_greedy/" + dataset_name + '/' + dataset_name + '_' + obj + '_' + weight_update + '_' +
        #     weight_update_method + '_realkd_col_' + str(col) + '_' + 'rep' + str(m) + ".txt", "w")
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
        scores = {}
        for r in regs:
            print('--------', r, '--------')
            fc_estimator.set_reg(r)
            scores[r] = cv(train, train_target, fc_estimator, labels, loss=loss)
        print('fc scores:', scores)
        # find best lambda
        reg = list(scores.keys())[0]
        for r in scores:
            if scores[r] < scores[reg]:
                reg = r
        selected_regs.append(reg)
        fc_estimator.set_reg(reg)
        try:
            fc_estimator.fit(train_df, train_sr)
            print(fc_estimator.rules_)
            for fc_ensemble in fc_estimator.history:
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
                print('coverage', coverage)
                print('train_risk', train_risk, 'test_risk', test_risk)
            fc_coverages_all.append(fc_coverages)
            fc_train_risk_all.append(sum(fc_train_risk) / len(fc_train_risk))
            fc_test_risk_all.append(sum(fc_test_risk) / len(fc_test_risk))

        except Exception as e:
            print('Error2: ', e)
        print('========= orth ===========')
        orth_estimator = GeneralRuleBoostingEstimator(num_rules=max_rule_num,
                                                      max_col_attr=col, search='exhaustive',
                                                      objective_function=objs['orth'],
                                                      weight_update_method=weight_upds['fc'](solver='Newton-CG'),
                                                      loss=loss)
        orth_estimator.set_reg(reg)
        orth_coverages.append(fc_coverages[0])
        for ensemble in fc_estimator.history:
            try:
                if len(ensemble) == 10:
                    break
                orth_estimator.rules_ = ensemble
                orth_estimator.num_rules = len(ensemble) + 1
                orth_estimator.fit(train_df, train_sr, has_origin_rules=True)
                orth_ensemble = orth_estimator.rules_
                orth_risk = sum(loss_func(train_sr, orth_ensemble(train_df))) / n + reg * sum(
                    [rule.y * rule.y for rule in orth_ensemble.members]) / 2 / n
                orth_test_risk = sum(loss_func(test_sr, orth_ensemble(test_df))) / len(test_sr)
                orth_train_risk = sum(loss_func(train_sr, orth_ensemble(train_df))) / n
                orth_fc_test_risk.append(orth_test_risk)
                orth_fc_train_risk.append(orth_train_risk)
                orth_fc_risk.append(orth_risk)
                orth_fc_ensembles.append(str(orth_ensemble))
                orth_coverage = sum(orth_ensemble[-1].q(train_df))
                orth_coverages.append(orth_coverage)
                print(orth_ensemble)
                print('risk', orth_risk)
                print('coverage', orth_coverage)
                print('train_risk', orth_train_risk, 'test_risk', orth_test_risk)
            except Exception as e:
                print('Error: ', e)
                break
        orth_coverages_all.append(orth_coverages)
        # try:
        #     for i in range(max_rule_num):
        #         output.write('\n=======iteration ' + str(i) + '========\n')
        #         if i < len(fc_risk):
        #             output.write('\nfc risk: ' + str(fc_risk[i]) + '\n')
        #             output.write('fc train risk: ' + str(fc_train_risk[i]) + '\n')
        #             output.write('fc test risk: ' + str(fc_test_risk[i]) + '\n')
        #             output.write(fc_ensembles[i])
        #     output.write('background rule: ' + str(log(max_y)) + ' if True')
        # except Exception as e:
        #     print('Error6: ', e)
        # output.write(str(selected_regs))
        # output.close()
    return fc_train_risk_all, fc_test_risk_all, fc_coverages_all, orth_coverages_all


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    res = {}
    for obj in ['mwg']:
        wupds = ['boosting'] if obj != 'xgb' else ['boosting', 'fc', 'keep']
        for weight_upd in wupds:
            upd_methods = ['GD', 'Newton-CG'] if weight_upd == 'fc' else ['']
            for upd in upd_methods:
                for col in [10]:
                    try:
                        res['vaccine' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                            evaluate('vaccine', '../datasets/counting_data/vaccination-data.csv',
                                     ['Population', 'WHO_REGION', 'NUMBER_VACCINES_TYPES_USED'],
                                     [int, int, int],
                                     'TOTAL_VACCINATIONS', target_type=int,
                                     obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                     loss='poisson', repeat=5, max_rule_num=10,
                                     regs=[0], col=col)
                    except Exception as e:
                        print("Error 1", e)
                    try:
                        res['ships' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                            evaluate('ships', '../datasets/count_data2/ships.csv',
                                     ['type', 'Construction', 'operation', 'months'], [int, int, int, int],
                                     'damage',
                                     target_type=int, loss='poisson', repeat=5, max_rule_num=10,
                                     obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                     regs=[0], col=col)
                    except Exception as e:
                        print("Error 1", e)
                    try:
                        res['smoking' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                            evaluate('smoking', '../datasets/count_data2/smoking.csv',
                                     ['age', 'smoke'], [int, int],
                                     'dead_per_100',
                                     target_type=float, loss='poisson', repeat=5, max_rule_num=10,
                                     obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                     regs=[0], col=col)
                    except Exception as e:
                        print("Error 1", e)
                    try:
                        res['covid_vic' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                            evaluate('covid_vic',
                                     '../datasets/counting_data/COVID19 Data Viz LGA Data - lga.csv',
                                     ['population', 'active', 'cases', 'band'], [int, int, int, int],
                                     'new', target_type=int,
                                     loss='poisson', repeat=5, max_rule_num=10,
                                     obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                     regs=[0], col=col)
                    except Exception as e:
                        print("Error 1", e)
                    try:
                        res['covid_world' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                            evaluate('covid_world',
                                     '../datasets/counting_data/WHO-COVID-19-global-table-data.csv',
                                     ['Population', 'WHO_Region', 'Cases - cumulative total'],
                                     [int, int, int],
                                     'Cases - newly reported in last 24 hours', target_type=int,
                                     loss='poisson', repeat=5, max_rule_num=10,
                                     obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                     regs=[0], col=col)
                    except Exception as e:
                        print("Error 1", e)
                    try:
                        res['covid' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                            evaluate('covid', '../datasets/covid/COVID-19 Coronavirus.csv',
                                     ['Population', 'Continent'],
                                     [int, int],
                                     'TotalDeaths/1M pop',
                                     target_type=int, loss='poisson', repeat=5, max_rule_num=10,
                                     obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                     regs=[0], col=col)
                    except Exception as e:
                        print("Error 1", e)
                    try:
                        res['bicycle' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                            evaluate('bicycle', '../datasets/bicycle_counts/Book1.csv',
                                     ['Day', 'High Temp (F)', 'Low Temp (F)', 'Precipitation'],
                                     [int, float, float, float],
                                     'Total',
                                     target_type=int, loss='poisson', repeat=5, max_rule_num=10,
                                     obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                     regs=[0], col=col)
                    except Exception as e:
                        print("Error 1", e)
                    try:
                        res['austin-water' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                            evaluate('austin-water',
                                     '../datasets/counting_data/AustinWater/' +
                                     'austin-water-wastewater-service-connection-count-by-zip-code-2.csv',
                                     ['PostCode', 'CustomerClass'], [int, int], 'ServiceConnections',
                                     target_type=int,
                                     loss='poisson', repeat=5, max_rule_num=10,
                                     obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                     regs=[0], col=col)
                    except Exception as e:
                        print("Error 1", e)
                    print(res)
