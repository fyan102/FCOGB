import os
import warnings

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_diabetes

from evaluation.cross_validation import cv
from evaluation.data_info import get_splits
from evaluation.data_preprocess import preprocess_datasets
from realkd.boosting import GeneralRuleBoostingEstimator, FullyCorrective, GradientBoostingObjectiveMWG, LineSearch, \
    GradientBoostingObjectiveGPE, KeepWeight, OrthogonalBoostingObjective
from realkd.rules import loss_function, GradientBoostingObjective

folder = '../experiment_output_coverage/'
objs = {'xgb': GradientBoostingObjective, 'mwg': GradientBoostingObjectiveMWG, 'gpe': GradientBoostingObjectiveGPE,
        'orth': OrthogonalBoostingObjective}
weight_upds = {'boosting': LineSearch, 'fc': FullyCorrective, 'keep': KeepWeight}


def evaluate(dataset_name, load_method, obj='xgb',
             weight_update='fc', weight_update_method='Newton-CG', feature_map={}, loss='squared', search='exhaustive',
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
                                                    max_col_attr=col, search='exhaustive' if obj=='xgb' else 'greedy',
                                                    objective_function=obj_function,
                                                    weight_update_method=weight_update_func,
                                                    loss=loss)

        if not os.path.exists(folder):
            os.makedirs(folder)
        if not os.path.exists(folder + "/" + dataset_name):
            os.makedirs(folder + "/" + dataset_name)
        output = open(
            folder + dataset_name + '/' + dataset_name + '_' + obj + '_' + weight_update + '_' +
            weight_update_method + '_realkd_col_' + str(col) + '_' + 'rep' + str(m) + ".txt", "w")
        train, test, train_target, test_target, _, _, _, n, labels = preprocess_datasets(load_method,
                                                                                         feature_map=feature_map,
                                                                                         random_seed=seeds[m])
        train_df = pd.DataFrame(train, columns=labels)
        test_df = pd.DataFrame(test, columns=labels)
        train_sr = pd.Series(train_target)
        test_sr = pd.Series(test_target)
        scores = {}
        if len(regs) == 1:
            reg = regs[0]
        else:
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
                test_risk = sum(loss_func(test_sr, fc_ensemble(test_df))) / len(test_sr)
                train_risk = sum(loss_func(train_sr, fc_ensemble(train_df))) / n
                fc_test_risk.append(test_risk)
                fc_train_risk.append(train_risk)
                fc_risk.append(risk)
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
        try:
            for i in range(max_rule_num - 1):
                output.write('\n=======iteration ' + str(i) + '========\n')
                if i < len(fc_risk):
                    output.write('\nfc risk: ' + str(fc_risk[i]) + '\n')
                    output.write('fc train risk: ' + str(fc_train_risk[i]) + '\n')
                    output.write('fc test risk: ' + str(fc_test_risk[i]) + '\n')
                    output.write('coverage: ' + str(fc_coverages[i]) + '\n')
                    output.write(fc_ensembles[i] + '\n')
                    output.write('orth \n')
                    output.write('\north risk: ' + str(orth_fc_risk[i]) + '\n')
                    output.write('fc train risk: ' + str(orth_fc_train_risk[i]) + '\n')
                    output.write('fc test risk: ' + str(orth_fc_test_risk[i]) + '\n')
                    output.write('coverage: ' + str(orth_coverages[i]) + '\n')
                    output.write(orth_fc_ensembles[i] + '\n')

        except Exception as e:
            print('Error6: ', e)
        output.write(str(selected_regs))
        output.close()
    return fc_train_risk_all, fc_test_risk_all, fc_coverages_all, orth_coverages_all


def eveluate_loaded():
    warnings.filterwarnings('ignore')
    res = {}
    for obj in ['mwg', 'gpe', 'xgb']:
        wupds = ['boosting'] if obj != 'xgb' else ['keep']
        for weight_upd in wupds:
            upd_methods = ['Newton-CG'] if weight_upd == 'fc' else ['']

            for upd in upd_methods:

                for col in [10]:  # finished
                    try:
                        res['iris' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                            evaluate('iris', load_iris, feature_map={'target': {0: -1, 1: 1, 2: -1}},
                                     repeat=5, regs=[0.1], loss='logistic',
                                     obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                     max_rule_num=10, col=col)
                    except Exception as e:
                        print("Error 1", e)
                for col in [5]:  # finished
                    try:
                        res['diabetes' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                            evaluate('load_diabetes', load_diabetes,
                                     repeat=5, regs=[0],
                                     obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                     max_rule_num=10, col=col)
                    except Exception as e:
                        print("Error 1", e)
                for col in [5]:
                    try:
                        res['breast' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                            evaluate('breast_cancer', load_breast_cancer, feature_map={'target': {0: -1, 1: 1}},
                                     repeat=5, regs=[0.1], loss='logistic',
                                     obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                     max_rule_num=10, col=col)
                    except Exception as e:
                        print("Error 1", e)
                for col in [10]:  # finished
                    try:
                        res['load_wine' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                            evaluate('load_wine', load_wine,
                                     feature_map={'target': {0: -1, 1: 1, 2: -1}},
                                     repeat=5, regs=[0.1],
                                     loss='logistic', obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                     max_rule_num=10, col=col)
                    except Exception as e:
                        print("Error 1", e)
    print(res)
