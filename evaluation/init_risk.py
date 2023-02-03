from math import log, exp
import numpy as np
import pandas as pd
from realkd.rules import loss_function

from evaluation.data_info import get_splits
from evaluation.data_preprocess import preprocess_pd, preprocess_gen
import warnings
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3


def calculate_risks(dataset_name, path, labels, feature_types, target, target_type=int, feature_map={}, loss='squared',
                    repeat=5):
    print('==========', dataset_name, '===========')
    print('---------------------------------------')
    seeds = get_splits()[dataset_name]
    fc_train_risk_all = []
    # fc_test_risk_all = []
    for m in range(repeat):
        loss_func = loss_function(loss)
        train, test, train_target, test_target, _, _, _, n = preprocess_pd(path,
                                                                           labels,
                                                                           feature_types,
                                                                           target, target_type=target_type,
                                                                           feature_map=feature_map,
                                                                           random_seed=seeds[m])
        ys = train_target + test_target
        train_sr = pd.Series(ys)
        # test_sr = pd.Series(test_target)
        # test_risk = sum(loss_func(np.zeros_like(test_sr), test_sr)) / len(test_sr)
        train_risk = sum(loss_func(np.zeros_like(train_sr), train_sr)) / len(train_sr)
        # fc_test_risk_all.append(test_risk)
        fc_train_risk_all.append(train_risk)
        print('train_risk', train_risk)
    return sum(fc_train_risk_all) / len(fc_train_risk_all)  # , sum(fc_test_risk_all)/len(fc_test_risk_all)


def gen_friedman(func_name, n, noise, random_seed, d=4):
    func_map = {'make_friedman1': make_friedman1, 'make_friedman2': make_friedman2, 'make_friedman3': make_friedman3, }
    if func_name == 'make_friedman1':
        x, y = func_map[func_name](n_samples=n, n_features=d, noise=noise, random_state=random_seed)
    else:
        x, y = func_map[func_name](n_samples=n, noise=noise, random_state=random_seed)
    labels = ['x' + str(i) for i in range(1, d + 1)]
    return x, y, labels


def calculate_risks_friedman(dataset_name, number, noise, d=4, test_size=0.2, loss='squared',
                             repeat=5):
    print('==========', dataset_name, '===========')
    print('---------------------------------------')
    seeds = get_splits()[dataset_name]
    fc_train_risk_all = []
    # fc_test_risk_all = []
    for m in range(repeat):
        loss_func = loss_function(loss)
        x, y, labels = gen_friedman(dataset_name, number, noise, 1000, d=d)
        train, test, train_target, test_target, _, _, _, n = preprocess_gen(x, y, test_size=test_size,
                                                                            random_seed=seeds[m])
        ys = train_target.tolist() + test_target.tolist()
        print(type(ys))
        train_sr = pd.Series(ys)
        # test_sr = pd.Series(test_target)
        # test_risk = sum(loss_func(np.zeros_like(test_sr), test_sr)) / len(test_sr)
        train_risk = sum(loss_func(np.zeros_like(train_sr), train_sr)) / len(train_sr)
        # fc_test_risk_all.append(test_risk)
        fc_train_risk_all.append(train_risk)
        print('train_risk', train_risk)
    return sum(fc_train_risk_all) / len(fc_train_risk_all)  # , sum(fc_test_risk_all)/len(fc_test_risk_all)


def calculate_risks_poisson(dataset_name, path, labels, feature_types, target, target_type=int, feature_map={},
                            loss='poisson',
                            repeat=5):
    print('==========', dataset_name, '===========')
    print('---------------------------------------')
    seeds = get_splits()[dataset_name]
    fc_train_risk_all = []
    fc_test_risk_all = []
    for m in range(repeat):
        loss_func = loss_function(loss)
        train, test, train_target, test_target, _, _, _, n = preprocess_pd(path,
                                                                           labels,
                                                                           feature_types,
                                                                           target, target_type=target_type,
                                                                           feature_map=feature_map,
                                                                           random_seed=seeds[m])
        ys = train_target + test_target
        train_sr = pd.Series(ys)
        # test_sr = pd.Series(test_target)
        logys = [log(y) if y != 0 else 0 for y in ys]
        max_y = exp(sum(logys) / len(ys))
        # test_risk = sum(loss_func(test_sr, np.ones_like(test_sr)*log(max_y))) / len(test_sr)
        train_risk = sum(loss_func(train_sr, np.ones_like(train_sr) * log(max_y))) / len(train_sr)
        # fc_test_risk_all.append(test_risk)
        fc_train_risk_all.append(train_risk)
        print('train_risk', train_risk)
    return sum(fc_train_risk_all) / len(fc_train_risk_all)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    res = {}
    res['gdp'] = \
        calculate_risks('gdp',
                        '../datasets/gdp_vs_satisfaction/GDP_vs_Satisfaction.csv',
                        ['GDP'], [int], 'Satisfaction', target_type=float,
                        feature_map={}, loss='squared',
                        repeat=5)
    res['titanic'] = \
        calculate_risks('titanic', '../datasets/titanic/train.csv',
                        ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
                        [int, str, float, int, int, float, str], 'Survived', target_type=int,
                        feature_map={'Sex': {'male': 1, 'female': 0},
                                     'Embarked': {'S': 1, 'C': 2, 'Q': 3},
                                     'Survived': {'0': -1, '1': 1}}, loss='logistic',
                        repeat=5)
    res['wage'] = \
        calculate_risks('wage', '../datasets/wages_demographics/wages.csv',
                        ['height', 'sex', 'race', 'ed', 'age'], [float, str, str, int, int], 'earn',
                        target_type=float,
                        feature_map={'sex': {'male': 1, 'female': 0},
                                     'race': {'white': 1, 'black': 2, 'hispanic': 3, 'other': 4}})
    res['insurance'] = \
        calculate_risks('insurance', '../datasets/insurance/insurance.csv',
                        ['age', 'sex', 'bmi', 'children', 'smoker', 'region'],
                        [int, str, float, int, str, str], 'charges', target_type=float,
                        feature_map={'sex': {'male': 1, 'female': 0},
                                     'smoker': {'yes': 1, 'no': 0},
                                     'region': {'southwest': 1, 'southeast': 2, 'northwest': 3,
                                                'northeast': 4}}
                        )
    res['world_happiness_indicator'] \
        = calculate_risks('world_happiness_indicator',
                          '../datasets/world_happiness_indicator/2019.csv',
                          ['GDP per capita', 'Social support',
                           'Healthy life expectancy',
                           'Freedom to make life choices',
                           'Generosity', 'Perceptions of corruption'],
                          [float, float, float, float, float, float, ],
                          'Score',
                          target_type=float,
                          repeat=5)

    res['IBM_HR'] = \
        calculate_risks('IBM_HR', '../datasets/IBM_HR/WA_Fn-UseC_-HR-Employee-Attrition.csv',
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
                            "Attrition": {'Yes': 1, 'No': 0},
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
                        loss='logistic', repeat=5, )
    res['used_cars'] = \
        calculate_risks('used_cars',
                        '../datasets/used_cars/cnt_km_year_powerPS_minPrice_maxPrice_avgPrice_sdPrice.csv',
                        ['count', 'km', 'year', 'powerPS'], [int, int, int, int], 'avgPrice',
                        target_type=float)
    res['vaccine'] = \
        calculate_risks_poisson('vaccine', '../datasets/counting_data/vaccination-data.csv',
                                ['Population', 'WHO_REGION', 'NUMBER_VACCINES_TYPES_USED'],
                                [int, int, int],
                                'TOTAL_VACCINATIONS', target_type=int,
                                )

    res['ships'] = \
        calculate_risks_poisson('ships', '../datasets/count_data2/ships.csv',
                                ['type', 'Construction', 'operation', 'months'], [int, int, int, int],
                                'damage',
                                target_type=int, loss='poisson', repeat=5)
    res['smoking'] = \
        calculate_risks_poisson('smoking', '../datasets/count_data2/smoking.csv',
                                ['age', 'smoke'], [int, int],
                                'dead_per_100',
                                target_type=float, loss='poisson', repeat=5)
    res['covid_vic'] = \
        calculate_risks_poisson('covid_vic',
                                '../datasets/counting_data/COVID19 Data Viz LGA Data - lga.csv',
                                ['population', 'active', 'cases', 'band'], [int, int, int, int],
                                'new', target_type=int,
                                loss='poisson', repeat=5)
    res['covid_world'] = \
        calculate_risks_poisson('covid_world',
                                '../datasets/counting_data/WHO-COVID-19-global-table-data.csv',
                                ['Population', 'WHO_Region', 'Cases - cumulative total'],
                                [int, int, int],
                                'Cases - newly reported in last 24 hours', target_type=int,
                                loss='poisson', repeat=5)
    res['covid'] = \
        calculate_risks_poisson('covid', '../datasets/covid/COVID-19 Coronavirus.csv',
                                ['Population', 'Continent'],
                                [int, int],
                                'TotalDeaths/1M pop',
                                target_type=int, loss='poisson', repeat=5)
    res['bicycle'] = \
        calculate_risks_poisson('bicycle', '../datasets/bicycle_counts/Book1.csv',
                                ['Day', 'High Temp (F)', 'Low Temp (F)', 'Precipitation'],
                                [int, float, float, float],
                                'Total',
                                target_type=int, loss='poisson', repeat=5)
    res['austin-water'] = \
        calculate_risks_poisson('austin-water',
                                '../datasets/counting_data/AustinWater/' +
                                'austin-water-wastewater-service-connection-count-by-zip-code-2.csv',
                                ['PostCode', 'CustomerClass'], [int, int], 'ServiceConnections',
                                target_type=int,
                                loss='poisson', repeat=5)
    res['make_friedman2'] = \
        calculate_risks_friedman('make_friedman2', 10000, 0.1, test_size=0.8,
                 repeat=5)
    res['make_friedman3'] = \
        calculate_risks_friedman('make_friedman3', 5000, 0.1, test_size=0.8,
                 repeat=5)
    res['make_friedman1'] = \
        calculate_risks_friedman('make_friedman1', 2000, 0.1, d=10, test_size=0.8,
                 repeat=5)
    print(res)
