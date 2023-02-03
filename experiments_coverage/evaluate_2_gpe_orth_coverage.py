import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from pandas import qcut
from realkd.boosting import GeneralRuleBoostingEstimator, FullyCorrective, GradientBoostingObjectiveMWG, LineSearch, \
    GradientBoostingObjectiveGPE, KeepWeight, OrthogonalBoostingObjective
from realkd.rules import Rule, AdditiveRuleEnsemble, loss_function, XGBRuleEstimator, GradientBoostingObjective
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
             weight_update='fc', weight_update_method='Newton-CG', feature_map={}, loss='squared',search='exhaustive',
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
                                                    max_col_attr=col, search=search,
                                                    objective_function=obj_function,
                                                    weight_update_method=weight_update_func,
                                                    loss=loss)

        # output = open(
        #     "../output20221223realkd_cv_eval_no_reg_exhaustive/" + dataset_name + '/' + dataset_name + '_' + obj + '_' + weight_update + '_' +
        #     weight_update_method + '_realkd_col_' + str(col) + '_' + 'rep' + str(m) + ".txt", "w")
        train, test, train_target, test_target, _, _, _, n = preprocess_pd(path,
                                                                           labels,
                                                                           feature_types,
                                                                           target, target_type=target_type,
                                                                           feature_map=feature_map,
                                                                           random_seed=seeds[m])
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
        # try:
        #     for i in range(max_rule_num):
        #         output.write('\n=======iteration ' + str(i) + '========\n')
        #         if i < len(fc_risk):
        #             output.write('\nfc risk: ' + str(fc_risk[i]) + '\n')
        #             output.write('fc train risk: ' + str(fc_train_risk[i]) + '\n')
        #             output.write('fc test risk: ' + str(fc_test_risk[i]) + '\n')
        #             output.write(fc_ensembles[i])
        # except Exception as e:
        #     print('Error6: ', e)
        # output.write(str(selected_regs))
        # output.close()
    return fc_train_risk_all, fc_test_risk_all, fc_coverages_all, orth_coverages_all


if __name__ == '__main__':
    res = {}
    warnings.filterwarnings('ignore')
    for obj in ['mwg']:
        wupds = ['boosting'] if obj != 'xgb' else ['boosting', 'fc', 'keep']
        for weight_upd in wupds:
            upd_methods = ['Newton-CG', 'GD'] if weight_upd == 'fc' else ['']
            for upd in upd_methods:
                for col in [10]:  # finished
                    try:
                        res['world_happiness_indicator' + '_' + obj + '_' + weight_upd + '_' + upd] \
                            = evaluate('world_happiness_indicator',
                                       '../datasets/world_happiness_indicator/2019.csv',
                                       ['GDP per capita', 'Social support',
                                        'Healthy life expectancy',
                                        'Freedom to make life choices',
                                        'Generosity', 'Perceptions of corruption'],
                                       [float, float, float, float, float, float, ],
                                       'Score',
                                       target_type=float,
                                       obj=obj, weight_update=weight_upd,
                                       weight_update_method=upd,
                                       regs=[0],
                                       repeat=5, max_rule_num=10, col=col)
                    except Exception as e:
                        print("Error 1", e)
                for col in [2]:
                    try:
                        res['Demographics' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                            evaluate('Demographics', '../datasets/Demographics/Demographics1.csv',
                                     ['Sex', 'Marital', 'Age', 'Edu', 'Occupation', 'LivingYears',
                                      'Persons',
                                      'PersonsUnder18', 'HouseholderStatus',
                                      'TypeOfHome', 'Ethnic', 'Language'],
                                     [str, str, int, int, str, int, int, int, str, str, str, str],
                                     'AnnualIncome',
                                     target_type=int,
                                     obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                     feature_map={'Sex': {' Male': 1, ' Female': 0},
                                                  'Marital': {' Married': 1, '': 0,
                                                              ' Single, never married': 2,
                                                              ' Divorced or separated': 3,
                                                              ' Living together, not married': 4,
                                                              ' Widowed': 5},
                                                  'Occupation': {'': 0, ' Homemaker': 1,
                                                                 ' Professional/Managerial': 2,
                                                                 ' Student, HS or College': 3,
                                                                 ' Retired': 4, ' Unemployed': 5,
                                                                 ' Factory Worker/Laborer/Driver': 6,
                                                                 ' Sales Worker': 7,
                                                                 ' Clerical/Service Worker': 8,
                                                                 ' Military': 9},
                                                  'HouseholderStatus': {'': 0, ' Own': 1, ' Rent': 2,
                                                                        ' Live with Parents/Family': 3},
                                                  'TypeOfHome': {'': 0, ' House': 1,
                                                                 ' Apartment': 2,
                                                                 ' Condominium': 3,
                                                                 ' Mobile Home': 4, ' Other': 5, },
                                                  'Ethnic': {'': 0, ' White': 1,
                                                             ' Hispanic': 2,
                                                             ' Asian': 3,
                                                             ' Black': 4, ' East Indian': 5,
                                                             ' Pacific Islander': 6,
                                                             ' American Indian': 7,
                                                             ' Other': 8, },
                                                  'Language': {'': 0, ' English': 1, ' Spanish': 2,
                                                               ' Other': 3, }
                                                  }, regs=[0],
                                     repeat=5, max_rule_num=10, col=col)
                    except Exception as e:
                        print("Error 1", e)

                for col in [3]:
                    try:
                        res['IBM_HR' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                            evaluate('IBM_HR', '../datasets/IBM_HR/WA_Fn-UseC_-HR-Employee-Attrition.csv',
                                     ["Age", 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
                                      'Education',
                                      'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
                                      'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
                                      'MaritalStatus',
                                      'MonthlyIncome',
                                      'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
                                      'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours',
                                      'StockOptionLevel',
                                      'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
                                      'YearsAtCompany',
                                      'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'],
                                     [int, str, int, str, int, int, str, int, str, int, int, int, str, int, str,
                                      int, int,
                                      int, str, int, int, int, int, int, int, int, int, int, int, int, int],
                                     'Attrition', target_type=str,
                                     feature_map={
                                         "BusinessTravel": {'Travel_Rarely': 1, 'Travel_Frequently': 2,
                                                            'Non-Travel': 3},
                                         "Attrition": {'Yes': 1, 'No': -1},
                                         'Department': {'Sales': 1, 'Research & Development': 2,
                                                        'Human Resources': 3},
                                         'EducationField': {'Life Sciences': 1, 'Medical': 2, 'Marketing': 3,
                                                            'Technical Degree': 4, 'Human Resources': 5,
                                                            'Other': 6},
                                         'Gender': {'Male': 1, 'Female': 0},
                                         'JobRole': {'Sales Executive': 1, 'Research Scientist': 2,
                                                     'Laboratory Technician': 3,
                                                     'Manufacturing Director': 4,
                                                     'Healthcare Representative': 5,
                                                     'Manager': 6, 'Human Resources': 7, 'Research Director': 8,
                                                     'Sales Representative': 9},
                                         'MaritalStatus': {'Single': 1, 'Married': 2, 'Divorced': 3},
                                         'OverTime': {'Yes': 1, 'No': -1}, },
                                     obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                     loss='logistic', repeat=5, regs=[0.1],
                                     max_rule_num=10, col=col)
                    except Exception as e:
                        print("Error 1", e)
    print(res)
