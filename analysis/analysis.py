import numpy as np


def get_max_risk(dataset_name):
    d = {'gdp': 43.986571428571416,
         'wage': 2029057842.6515312,
         'insurance': 322646873.58870494,
         'used_cars': 194464336.77843493,
         'boston': 592.1469169960473,
         'world_happiness_indicator': 30.46778212179487,
         'Demographics': 31.64835409252669,
         'mobile_prices': 3.5,
         'load_diabetes': 29074.481900452487,
         'suicide_rates_cleaned': 523.7783290690134,
         'videogamesales': 2.70680865164444,
         'red_wine_quality': 32.41651031894934,
         'who_life_expectancy': 4882.764241803283,
         'vaccine': 228506505.50612897,
         'ships': 12.342277392898044,
         'smoking': 7.500299349437478,
         'covid_vic': 51.62076805721374,
         'covid_world': 14143.401514056353,
         'covid': 1082.8170359400497,
         'bicycle': 1076.416963882381,
         'austin-water': 3975.767978495398,
         'make_friedman2': 379683.4049960418,
         'make_friedman3': 1.8780821278399789,
         'make_friedman1': 226.89125115757616
         }
    if dataset_name in d:
        return d[dataset_name]
    else:
        return 1


def get_bg_risk(dataset_name):
    d = {'gdp': 0.5252423469387753,
         'titanic': 0.9660665186125368,
         'wage': 986833103.1037899,
         'insurance': 148541844.24274337,
         'world_happiness_indicator': 1.2511818139308006,
         'Demographics': 7.680118972030896,
         'IBM_HR': 0.6174512131279031,
         'iris': 0.9101532032789194,
         'load_diabetes': 5789.639210497658,
         'breast_cancer': 0.9466414596942435,
         'used_cars': 87247418.0776172,
         'tic-tac-toe': 0.9304535871919674,
         'boston': 86.84682804273004,
         'make_friedman2': 143009.26234621546,
         'make_friedman3': 0.09655585795040648,
         'make_friedman1': 22.876431760541482,
         'load_wine': 0.9589205062979972,
         'banknote': 0.993383415922522,
         'liver': 0.9848666654761133,
         'magic': 0.9352015687833055,
         'adult': 0.8096813325498946,
         'GenderRecognition': 0.9998202608432892,
         'mobile_prices': 1.254624823328245,
         'telco_churn': 0.8340345526094785,
         'who_life_expectancy': 77.21309127435023,
         'digits5': 0.4741918513355029,
         'suicide_rates_cleaned': 362.13148872147474,
         'videogamesales': 2.5801642402023517,
         'red_wine_quality': 0.6471216208906223}
    return d[dataset_name] if dataset_name in d else 1


def get_risks_from_file(dataset_name, algo_name,
                        location='../experiment_output_20231221varreg_new_no_bg_ruleexhaustive',
                        repeat=0, col=10, max_length=30, max_components=51):
    train_risks = np.array([])
    test_risks = np.array([])
    num_components = np.array([])
    file_name = dataset_name + '_' + algo_name + '_realkd_col_' + str(col) + '_rep' + str(repeat) + '.txt'
    file_path = location + '/' + dataset_name + '/' + file_name
    try:
        f = open(file_path, 'r')
    except Exception as e:
        print(e)
        print(file_path, 'does not exist')
        return train_risks, test_risks, num_components
    j = -1
    for line in f:
        if 'iteration' in line:
            j += 1
            if j != 0:
                num_components = np.append(num_components, components)
                if components >= max_components:
                    break
            components = 0
        if j >= max_length:
            break
        words = line.split(' ')
        if 'fc train risk' in line:
            train_risks = np.append(train_risks, float(words[-1]))
        elif 'fc test risk' in line:
            test_risks = np.append(test_risks, float(words[-1]))
        if 'if' in line:
            components += line.count('=') + 1
    return train_risks, test_risks, num_components


def get_components_from_file(dataset_name, algo_name,
                             location='../experiment_output_20231221varreg_new_no_bg_ruleexhaustive',
                             repeat=0, col=10, max_length=30, max_components=51):
    train_risks = np.array([])
    test_risks = np.array([])
    num_components = np.array([])
    file_name = dataset_name + '_' + algo_name.replace('*', '') + '_realkd_col_' + str(col) + '_rep' + str(
        repeat) + '.txt'
    file_path = location + '/' + dataset_name + '/' + file_name
    try:
        f = open(file_path, 'r')
    except Exception as e:
        # print(e)
        # print(file_path, 'does not exist')
        return num_components, 0
    j = -1
    components = 0
    for line in f:
        if 'iteration' in line:
            j += 1
            # components = 0
            if j != 0:
                num_components = np.append(num_components, components)
                # if components >= 50:
                #     break
            components = 0
        if j >= max_length:
            break
        if 'if' in line:
            components += line.count('=') + 1
    num_components = np.append(num_components, components)
    return num_components, j


def map_risk_components(dataset_name, risks, components, max_components=31):
    num_components = np.array([])
    risk_values = np.array([])
    print(risks)
    risk = get_max_risk(dataset_name)
    j = 0
    risk_values = np.append(risk_values, risk)
    num_components = np.append(num_components, 1)
    j += 1
    poisson_datasets = ['covid_vic', 'covid', 'bicycle', 'ships', 'smoking']
    # else:
    #     j = 0
    if dataset_name not in poisson_datasets:
        risk = get_bg_risk(dataset_name)
    for i in range(len(components)):
        while j < min(components[i], max_components) - (1 if dataset_name in poisson_datasets else 0):
            num_components = np.append(num_components, j)
            risk_values = np.append(risk_values, risk)
            j += 1
        risk = risks[i]
    while j < max_components:
        num_components = np.append(num_components, j)
        risk_values = np.append(risk_values, risks[-1])
        j += 1
    return num_components, risk_values


def get_all_risks(dataset_name, algo_name, location='../experiment_output_20231221varreg_new_no_bg_ruleexhaustive',
                  repeat=1, col=10, max_length=30, max_components=51):
    train_risks = np.array([])
    test_risks = np.array([])
    num_components = np.array([])
    # if "*" in algo_name:
    #     location='../experiment_output_greedy'
    #     # algo_name=algo_name.replace("*",'')
    # #     print(algo_name.replace('*', ''))
    train = np.array([0] * max_components)
    test = np.array([0] * max_components)
    num = np.array([0] * max_components)
    for i in range(repeat):
        try:
            train, test, num = get_risks_from_file(dataset_name, algo_name.replace('*', ''), location=location,
                                                   repeat=i,
                                                   col=col,
                                                   max_length=max_length, max_components=max_components)
            print(train, test, num)
        except Exception as e:
            print(e)
    components, train_risk_values = map_risk_components(dataset_name, train, num, max_components=max_components)
    _, test_risk_values = map_risk_components(dataset_name, test, num, max_components=max_components)
    train_risks = np.append(train_risks, train_risk_values)
    test_risks = np.append(test_risks, test_risk_values)
    num_components = np.append(num_components, components)
    return np.reshape(train_risks, (-1, max_components)), np.reshape(test_risks, (-1, max_components)), \
           np.reshape(num_components, (-1, max_components))


def get_average_risks_for_datasets(dataset_name, algo_name,
                                   location='../experiment_output_20231221varreg_new_no_bg_ruleexhaustive',
                                   repeat=1, col=10, max_length=30, max_components=52):
    train, test, _ = get_all_risks(dataset_name, algo_name, location=location,
                                   repeat=repeat, col=col, max_length=max_length, max_components=max_components)
    train_avg = train.mean(axis=0)
    test_avg = test.mean(axis=0)
    return train_avg, test_avg

