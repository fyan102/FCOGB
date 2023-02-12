import warnings

from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_wine

from evaluation.evaluate_boosting import evaluate_dataset, evaluate_loaded_data, evaluate_friedman, evaluate_poisson


def start_experiments():
    all_experiments(search_methods=['exhaustive', 'greedy'], objectives=['orth'], weight_updates=['fc'])
    all_experiments(search_methods=['exhaustive'], objectives=['xgb'], weight_updates=['keep'])
    all_experiments(search_methods=['greedy'], objectives=['mwg'], weight_updates=['boosting'])
    all_experiments(search_methods=['exhaustive', 'greedy'], objectives=['gpe'], weight_updates=['boosting', 'fc'])


def all_experiments(search_methods=['exhaustive'], objectives=['orth'], weight_updates=['fc']):
    warnings.filterwarnings('ignore')
    res = {}
    for s in search_methods:
        for obj in objectives:
            for weight_upd in weight_updates:
                upd_methods = [''] if weight_upd != 'fc' else ['Newton-CG']
                for upd in upd_methods:
                    for col in [20]:
                        res['gdp' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                            evaluate_dataset('gdp', '../datasets/gdp_vs_satisfaction/GDP_vs_Satisfaction.csv',
                                             ['GDP'], [int], 'Satisfaction', target_type=float,
                                             obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                             feature_map={}, loss='squared',
                                             repeat=5, max_rule_num=10, search=s,
                                             regs=[0],
                                             col=col)
                    for col in [10]:
                        try:
                            res['titanic' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_dataset('titanic', '../datasets/titanic/train.csv',
                                                 ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
                                                 [int, str, float, int, int, float, str], 'Survived', target_type=int,
                                                 feature_map={'Sex': {'male': 1, 'female': 0},
                                                              'Embarked': {'S': 1, 'C': 2, 'Q': 3},
                                                              'Survived': {'0': -1, '1': 1}}, loss='logistic',
                                                 repeat=5, regs=[0.1], search=s,
                                                 obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                                 max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)
                        try:
                            res['wage' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_dataset('wage', '../datasets/wages_demographics/wages.csv',
                                         ['height', 'sex', 'race', 'ed', 'age'], [float, str, str, int, int], 'earn',
                                         target_type=float,
                                         feature_map={'sex': {'male': 1, 'female': 0},
                                                      'race': {'white': 1, 'black': 2, 'hispanic': 3, 'other': 4}},
                                         regs=[0], search=s,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                         repeat=5, max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)
                        try:
                            res['insurance' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_dataset('insurance', '../datasets/insurance/insurance.csv',
                                         ['age', 'sex', 'bmi', 'children', 'smoker', 'region'],
                                         [int, str, float, int, str, str], 'charges', target_type=float,
                                         feature_map={'sex': {'male': 1, 'female': 0},
                                                      'smoker': {'yes': 1, 'no': 0},
                                                      'region': {'southwest': 1, 'southeast': 2, 'northwest': 3,
                                                                 'northeast': 4}},
                                         regs=[0], search=s,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                         repeat=5, max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)
                    for col in [10]:  # finished
                        try:
                            res['world_happiness_indicator' + '_' + obj + '_' + weight_upd + '_' + upd] \
                                = evaluate_dataset('world_happiness_indicator',
                                                   '../datasets/world_happiness_indicator/2019.csv',
                                                   ['GDP per capita', 'Social support',
                                                    'Healthy life expectancy',
                                                    'Freedom to make life choices',
                                                    'Generosity', 'Perceptions of corruption'],
                                                   [float, float, float, float, float, float, ],
                                                   'Score',
                                                   target_type=float, search=s,
                                                   obj=obj, weight_update=weight_upd,
                                                   weight_update_method=upd,
                                                   regs=[0],
                                                   repeat=5, max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)
                    for col in [2]:
                        try:
                            res['Demographics' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_dataset('Demographics', '../datasets/Demographics/Demographics1.csv',
                                                 ['Sex', 'Marital', 'Age', 'Edu', 'Occupation', 'LivingYears',
                                                  'Persons',
                                                  'PersonsUnder18', 'HouseholderStatus',
                                                  'TypeOfHome', 'Ethnic', 'Language'],
                                                 [str, str, int, int, str, int, int, int, str, str, str, str],
                                                 'AnnualIncome',
                                                 target_type=int, search=s,
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
                                                              'HouseholderStatus': {'': 0, ' Own': 1,
                                                                                    ' Rent': 2,
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
                                evaluate_dataset('IBM_HR',
                                                 '../datasets/IBM_HR/WA_Fn-UseC_-HR-Employee-Attrition.csv',
                                                 ["Age", 'BusinessTravel', 'DailyRate', 'Department',
                                                  'DistanceFromHome',
                                                  'Education',
                                                  'EducationField', 'EnvironmentSatisfaction', 'Gender',
                                                  'HourlyRate',
                                                  'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
                                                  'MaritalStatus',
                                                  'MonthlyIncome',
                                                  'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
                                                  'PercentSalaryHike',
                                                  'PerformanceRating', 'RelationshipSatisfaction',
                                                  'StandardHours',
                                                  'StockOptionLevel',
                                                  'TotalWorkingYears', 'TrainingTimesLastYear',
                                                  'WorkLifeBalance',
                                                  'YearsAtCompany',
                                                  'YearsInCurrentRole', 'YearsSinceLastPromotion',
                                                  'YearsWithCurrManager'],
                                                 [int, str, int, str, int, int, str, int, str, int, int, int,
                                                  str, int, str,
                                                  int, int,
                                                  int, str, int, int, int, int, int, int, int, int, int, int,
                                                  int, int],
                                                 'Attrition', target_type=str,
                                                 feature_map={
                                                     "BusinessTravel": {'Travel_Rarely': 1,
                                                                        'Travel_Frequently': 2,
                                                                        'Non-Travel': 3},
                                                     "Attrition": {'Yes': 1, 'No': -1},
                                                     'Department': {'Sales': 1, 'Research & Development': 2,
                                                                    'Human Resources': 3},
                                                     'EducationField': {'Life Sciences': 1, 'Medical': 2,
                                                                        'Marketing': 3,
                                                                        'Technical Degree': 4,
                                                                        'Human Resources': 5,
                                                                        'Other': 6},
                                                     'Gender': {'Male': 1, 'Female': 0},
                                                     'JobRole': {'Sales Executive': 1, 'Research Scientist': 2,
                                                                 'Laboratory Technician': 3,
                                                                 'Manufacturing Director': 4,
                                                                 'Healthcare Representative': 5,
                                                                 'Manager': 6, 'Human Resources': 7,
                                                                 'Research Director': 8,
                                                                 'Sales Representative': 9},
                                                     'MaritalStatus': {'Single': 1, 'Married': 2,
                                                                       'Divorced': 3},
                                                     'OverTime': {'Yes': 1, 'No': -1}, },
                                                 obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                                 loss='logistic', repeat=5, regs=[0.1], search=s,
                                                 max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)
                    for col in [10]:  # finished
                        # try:
                            res['iris' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_loaded_data('iris', load_iris, feature_map={'target': {0: -1, 1: 1, 2: -1}},
                                         repeat=5, regs=[0.1], loss='logistic', search=s,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                         max_rule_num=10, col=col)
                        # except Exception as e:
                        #     print("Error 1", e)
                    for col in [5]:  # finished
                        try:
                            res['diabetes' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_loaded_data('load_diabetes', load_diabetes,
                                         repeat=5, regs=[0], search=s,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                         max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)
                    for col in [5]:
                        try:
                            res['breast' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_loaded_data('breast_cancer', load_breast_cancer,
                                         repeat=5, regs=[0.1], loss='logistic',feature_map={'target': {0: -1, 1: 1}},
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd, search=s,
                                         max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)
                    for col in [10]:
                        try:
                            res['used_cars' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_dataset('used_cars',
                                         '../datasets/used_cars/cnt_km_year_powerPS_minPrice_maxPrice_avgPrice_sdPrice.csv',
                                         ['count', 'km', 'year', 'powerPS'], [int, int, int, int], 'avgPrice',
                                         target_type=float, regs=[0], search=s,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                         repeat=5, max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)
                        try:
                            res['tic_tac_toe' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_dataset('tic-tac-toe', '../datasets/tic_tac_toe/tic_tac_toe.csv',
                                         ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"],
                                         [str, str, str, str, str, str, str, str, str], 'V10', target_type=str,
                                         feature_map={'V1': {'x': 1, 'o': 2, 'b': 3},
                                                      'V2': {'x': 1, 'o': 2, 'b': 3},
                                                      'V3': {'x': 1, 'o': 2, 'b': 3},
                                                      'V4': {'x': 1, 'o': 2, 'b': 3},
                                                      'V5': {'x': 1, 'o': 2, 'b': 3},
                                                      'V6': {'x': 1, 'o': 2, 'b': 3},
                                                      'V7': {'x': 1, 'o': 2, 'b': 3},
                                                      'V8': {'x': 1, 'o': 2, 'b': 3},
                                                      'V9': {'x': 1, 'o': 2, 'b': 3},
                                                      'V10': {'positive': 1, 'negative': -1}},
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                         regs=[0.1], search=s,
                                         loss='logistic', repeat=5, max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)


                    for col in [4]:
                        try:
                            res['boston' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_dataset('boston', '../datasets/boston/boston_house_prices.csv',
                                         ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
                                          'B',
                                          'LSTAT'],
                                         [float, float, float, float, float, float, float, int, int, float, float,
                                          float],
                                         'MEDV', regs=[0], search=s,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                         target_type=float, repeat=5, max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)

                    for col in [10]:
                        try:
                            res['fried2' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_friedman('make_friedman2', 10000, 0.1, test_size=0.8,
                                                  repeat=5, regs=[0], search=s,
                                                  obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                                  max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)
                    for col in [10]:
                        try:
                            res['fried3' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_friedman('make_friedman3', 5000, 0.1, test_size=0.8,
                                                  repeat=5, regs=[0], search=s,
                                                  obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                                  max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)
                    for col in [4]:
                        try:
                            res['fried1' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_friedman('make_friedman1', 2000, 0.1, d=10, test_size=0.8,
                                                  repeat=5, regs=[0], search=s,
                                                  obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                                  max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)
                    for col in [6]:  #
                        res['load_wine' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                            evaluate_loaded_data('load_wine', load_wine,
                                                 feature_map={'target': {0: -1, 1: 1, 2: -1}},
                                                 repeat=5, regs=[0.1], search=s,
                                                 loss='logistic', obj=obj, weight_update=weight_upd,
                                                 weight_update_method=upd,
                                                 max_rule_num=10, col=col)
                    for col in [10]:
                        try:
                            res['ships' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_poisson('ships', '../datasets/count_data2/ships.csv',
                                         ['type', 'Construction', 'operation', 'months'], [int, int, int, int],
                                         'damage',
                                         target_type=int, loss='poisson', repeat=5, max_rule_num=10,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd, search=s,
                                         regs=[0], col=col)
                        except Exception as e:
                            print("Error 1", e)
                        try:
                            res['smoking' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_poisson('smoking', '../datasets/count_data2/smoking.csv',
                                         ['age', 'smoke'], [int, int],
                                         'dead_per_100',
                                         target_type=float, loss='poisson', repeat=5, max_rule_num=10,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd, search=s,
                                         regs=[0], col=col)
                        except Exception as e:
                            print("Error 1", e)
                        try:
                            res['covid_vic' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_poisson('covid_vic',
                                         '../datasets/counting_data/COVID19 Data Viz LGA Data - lga.csv',
                                         ['population', 'active', 'cases', 'band'], [int, int, int, int],
                                         'new', target_type=int,
                                         loss='poisson', repeat=5, max_rule_num=10,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd, search=s,
                                         regs=[0], col=col)
                        except Exception as e:
                            print("Error 1", e)
                        try:
                            res['covid' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_poisson('covid', '../datasets/covid/COVID-19 Coronavirus.csv',
                                         ['Population', 'Continent'],
                                         [int, int],
                                         'TotalDeaths/1M pop',
                                         target_type=int, loss='poisson', repeat=5, max_rule_num=10,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd, search=s,
                                         regs=[0], col=col)
                        except Exception as e:
                            print("Error 1", e)
                        try:
                            res['bicycle' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_poisson('bicycle', '../datasets/bicycle_counts/Book1.csv',
                                         ['Day', 'High Temp (F)', 'Low Temp (F)', 'Precipitation'],
                                         [int, float, float, float],
                                         'Total',
                                         target_type=int, loss='poisson', repeat=5, max_rule_num=10,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd, search=s,
                                         regs=[0], col=col)
                        except Exception as e:
                            print("Error 1", e)
                    for col in [10]:
                        try:
                            res['banknote' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_dataset('banknote', '../datasets/banknotes/banknote.csv',
                                         ['variance', 'skewness', 'curtosis', 'entropy'],
                                         [float, float, float, float], 'class', target_type=int,
                                         feature_map={'class': {'0': -1, '1': 1}}, loss='logistic',
                                         repeat=5, regs=[0.1], search=s,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                         max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)

                        try:
                            res['liver' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_dataset('liver', '../datasets/liver/liver.csv',
                                         ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks'],
                                         [int, int, int, int, int, float], 'selector', target_type=int,
                                         feature_map={'selector': {'1': 1, '2': -1}, },
                                         regs=[0.1], search=s, loss='logistic',
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                         repeat=2, max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)
                    for col in [5]:
                        try:
                            res['magic' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_dataset('magic', '../datasets/magic/magic04.csv',
                                         ['fLen1t-1', 'fWidt-1', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Lon1',
                                          'fM3Trans', 'fAlp-1a', 'fDist'],
                                         [float, float, float, float, float, float, float, float, float, float],
                                         'class',
                                         target_type=int, loss='logistic',
                                         regs=[0.1], search=s,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                         repeat=1, max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)
                    for col in [5]:
                        try:
                            res['adult' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_dataset('adult', '../datasets/adult/adult.csv',
                                         ['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status',
                                          'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                          'hours-per-week'],
                                         [int, str, int, int, str, str, str, int, int, int, int],
                                         'output', target_type=int,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                         feature_map={
                                             'workclass': {'Private': 0, 'Self-emp-not-inc': 1, 'Self-emp-inc': 2,
                                                           'Federal-gov': 3, 'Local-gov': 4, 'State-gov': 5,
                                                           'Without-pay': 6, 'Never-worked': 7},
                                             'marital-status': {'Married-civ-spouse': 0, 'Divorced': 1,
                                                                'Never-married': 2, 'Separated': 3, 'Widowed': 4,
                                                                'Married-spouse-absent': 5, 'Married-AF-spouse': 6},
                                             'relationship': {'Wife': 0, 'Own-child': 1, 'Husband': 2,
                                                              'Not-in-family': 3, 'Other-relative': 4, 'Unmarried': 5},
                                             'race': {'White': 0, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 2,
                                                      'Other': 3, 'Black': 4}
                                         },
                                         loss='logistic', search=s,
                                         repeat=1, max_rule_num=10,
                                         regs=[0.1],
                                         col=col)
                        except Exception as e:
                            print("Error 1", e)
                    for col in [3]:
                        try:
                            res['GenderRecognition' + '_' + obj + '_' + weight_upd + '_' + upd] = evaluate_dataset(
                                'GenderRecognition',
                                '../datasets/GenderRecognition/voice.csv',
                                ["meanfreq", "sd", "median", "Q25", "Q75", "IQR",
                                 "skew", "kurt",
                                 "sp.ent", "sfm", "mode", "centroid", "meanfun",
                                 "minfun", "maxfun",
                                 "meandom", "mindom", "maxdom", "dfrange", "modindx"],
                                [float, float, float, float, float, float, float, float,
                                 float, float,
                                 float, float, float, float, float, float, float, float,
                                 float, float],
                                "label", target_type=str,
                                feature_map={"label": {'male': 1, 'female': -1}},
                                obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                loss='logistic', search=s,
                                repeat=1, regs=[0.1],
                                max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)

                    for col in [4]:
                        try:
                            res['mobile_prices' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_dataset('mobile_prices',
                                         '../datasets/mobile_prices/train.csv',
                                         ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc',
                                          'four_g',
                                          'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc',
                                          'px_height',
                                          'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
                                          'touch_screen',
                                          'wifi'],
                                         [int, int, float, int, int, int, int, float, int, int, int,
                                          int, int, int,
                                          int,
                                          int, int, int, int, int, ], 'price_range',
                                         target_type=int,
                                         regs=[0], search=s,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                         repeat=1, max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)
                    for col in [4]:
                        try:
                            res['telco_churn' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_dataset('telco_churn',
                                         '../datasets/telco_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv',
                                         ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                                          'PhoneService', 'MultipleLines', 'InternetService',
                                          'OnlineSecurity',
                                          'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                          'StreamingTV',
                                          'StreamingMovies', 'Contract', 'PaperlessBilling',
                                          'PaymentMethod',
                                          'MonthlyCharges', 'TotalCharges', ],
                                         [str, int, str, str, int, str, str, str, str, str, str, str,
                                          str, str,
                                          str, str, str, float, float],
                                         'Churn',
                                         target_type=str,
                                         feature_map={'gender': {'Male': 1, 'Female': 0},
                                                      'Partner': {'Yes': 1, 'No': 0},
                                                      'Dependents': {'Yes': 1, 'No': 0},
                                                      'PhoneService': {'Yes': 1, 'No': 0},
                                                      'MultipleLines': {'Yes': 1, 'No': 2,
                                                                        'No phone service': 3},
                                                      'InternetService': {'DSL': 1, 'Fiber optic': 2,
                                                                          'No': 3},
                                                      'OnlineSecurity': {'Yes': 1, 'No': 2,
                                                                         'No internet service': 3},
                                                      'OnlineBackup': {'Yes': 1, 'No': 2,
                                                                       'No internet service': 3},
                                                      'DeviceProtection': {'Yes': 1, 'No': 2,
                                                                           'No internet service': 3},
                                                      'TechSupport': {'Yes': 1, 'No': 2,
                                                                      'No internet service': 3},
                                                      'StreamingTV': {'Yes': 1, 'No': 2,
                                                                      'No internet service': 3},
                                                      'StreamingMovies': {'Yes': 1, 'No': 2,
                                                                          'No internet service': 3},
                                                      'Contract': {'Month-to-month': 1, 'One year': 2,
                                                                   'Two year': 3, },
                                                      'PaperlessBilling': {'Yes': 1, 'No': 0},
                                                      'PaymentMethod': {'Electronic check': 1,
                                                                        'Mailed check': 2,
                                                                        'Bank transfer (automatic)': 3,
                                                                        'Credit card (automatic)': 4},
                                                      'Churn': {'Yes': 1, 'No': -1},
                                                      }, regs=[0.1], search=s,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                         repeat=1, max_rule_num=10, loss='logistic', col=col)
                        except Exception as e:
                            print("Error 1", e)
                    for col in [5]:
                        try:
                            res['who_life_expectancy' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_dataset('who_life_expectancy', '../datasets/who_life_expectancy/Life Expectancy Data.csv',
                                         ['Year', 'Status', 'Adult Mortality', 'infant deaths', 'Alcohol',
                                          'percentage expenditure',
                                          'Hepatitis B', 'Measles', 'BMI', 'under-five deaths', 'Polio',
                                          'Total expenditure',
                                          'Diphtheria', 'HIV/AIDS', 'GDP', 'Population', 'thinness  1-19 years',
                                          'thinness 5-9 years', 'Income composition of resources', 'Schooling'],
                                         [int, str, int, int, float, float, int, int, float, int, int, float, int, float,
                                          float, int, float, float, float, float],
                                         'Life expectancy', target_type=float,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                         feature_map={
                                             'Status': {'Developing': 0, 'Developed': 1}
                                         },
                                         loss='squared', search=s,
                                         repeat=1, max_rule_num=10,
                                         regs=[0],
                                         col=col)
                        except Exception as e:
                            print("Error 1", e)
                    for col in [3]:
                        try:
                            res['digits5' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_dataset('digits5', '../datasets/digits/digits.csv',
                                         ['pixel_' + str(i) + '_' + str(j) for j in range(8) for i in range(8)],
                                         [int] * 64,
                                         'target',
                                         feature_map={
                                             'target': {'5': 1, '0': -1, '1': -1, '2': -1, '3': -1, '4': -1, '6': -1,
                                                        '7': -1, '8': -1,
                                                        '9': -1}},
                                         target_type=int, loss='logistic',
                                         regs=[0.1], search=s,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                         repeat=1, max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)
                    for col in [3]:
                        try:
                            res['suicide_rates_cleaned' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_dataset('suicide_rates_cleaned', '../datasets/suicide_rates_cleaned/master.csv',
                                         ['year', 'sex', 'age', 'population', 'gdp_for_year ($)', 'gdp_per_capita ($)',
                                          'generation'],
                                         [int, int, int, int, float, float, int], 'suicides/100k pop', target_type=float,
                                         loss='squared',
                                         repeat=1, regs=[0], search=s,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                         max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)

                        try:
                            res['videogamesales' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_dataset('videogamesales', '../datasets/videogamesales/vgsales.csv',
                                         ['Platform', 'Year', 'Genre'],
                                         [int, int, int], 'Global_Sales', target_type=float,
                                         regs=[0], search=s, loss='squared',
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                         repeat=1, max_rule_num=10, col=col)
                        except Exception as e:
                            print("Error 1", e)
                    for col in [5]:
                        try:
                            res['red_wine_quality' + '_' + obj + '_' + weight_upd + '_' + upd] = \
                                evaluate_dataset('red_wine_quality', '../datasets/red_wine_quality/winequality-red.csv',
                                         ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                          'chlorides',
                                          'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates',
                                          'alcohol'],
                                         [float, float, float, float, float, float, float, float, float, float, float],
                                         'quality', target_type=float,
                                         obj=obj, weight_update=weight_upd, weight_update_method=upd,
                                         feature_map={},
                                         loss='squared', search=s,
                                         repeat=5, max_rule_num=10,
                                         regs=[0.1],
                                         col=col)
                        except Exception as e:
                            print("Error 1", e)
                print(res)
