import csv
from realkd.rules import loss_function
from sklearn.datasets import load_wine, load_iris, load_diabetes, load_breast_cancer


def calc_original_risk(filename, target, target_type, feature_map={}, loss='squared'):
    yy = []
    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile)
        y_index = -1
        first = True
        for row in spamreader:
            if first:
                # print(row)
                for i in range(len(row)):
                    if row[i] == target:
                        y_index = i
                first = False
            else:
                try:
                    yy.append(
                        target_type(row[y_index]) if target not in feature_map else feature_map[target][row[y_index]])
                except Exception as e:
                    print(e)
        risk = 0
        loss_func = loss_function(loss)
        for y in yy:
            risk += loss_func(y, 0)
        risk /= len(yy)
        data_point_number = len(yy)
        return risk, data_point_number


def calc_risk_load(load_dataset, feature_map={}, loss='squared'):
    dataset = load_dataset()
    yy = dataset.target
    if len(feature_map) != 0:
        for i in range(len(yy)):
            try:
                yy[i] = feature_map['target'][yy[i]]
            except Exception as e:
                print(e)
    risk = 0
    loss_func = loss_function(loss)
    for y in yy:
        risk += loss_func(y, 0)
    risk /= len(yy)
    return risk, len(yy)


if __name__ == '__main__':
    res = {}
    res['gdp'] = calc_original_risk('../datasets/gdp_vs_satisfaction/GDP_vs_Satisfaction.csv', 'Satisfaction',
                                    target_type=float)
    res['wage'] = calc_original_risk('../datasets/wages_demographics/wages.csv', 'earn',
                                     target_type=float, )
    res['titanic'] = calc_original_risk('../datasets/titanic/train.csv',
                                        'Survived', target_type=int,
                                        feature_map={'Survived': {'0': -1, '1': 1}}, loss='logistic')
    res['insurance'] = calc_original_risk('../datasets/insurance/insurance.csv', 'charges', target_type=float, )
    res['used_cars'] = calc_original_risk(
        '../datasets/used_cars/cnt_km_year_powerPS_minPrice_maxPrice_avgPrice_sdPrice.csv', 'avgPrice',
        target_type=float, )
    res['tic_tac_toe'] = calc_original_risk('../datasets/tic_tac_toe/tic_tac_toe.csv',
                                            'V10',
                                            target_type=str,
                                            feature_map={
                                                'V10': {'positive': 1, 'negative': -1}},
                                            loss='logistic')
    res['boston'] = calc_original_risk('../datasets/boston/boston_house_prices.csv', 'MEDV',
                                       target_type=float)
    res['world_happiness_indicator'] = calc_original_risk('../datasets/world_happiness_indicator/2019.csv', 'Score',
                                                          target_type=float, )
    res['Demographics'] = calc_original_risk('../datasets/Demographics/Demographics1.csv', 'AnnualIncome',
                                             target_type=int, )
    res['IBM_HR'] = calc_original_risk('../datasets/IBM_HR/WA_Fn-UseC_-HR-Employee-Attrition.csv',
                                       'Attrition', target_type=str,
                                       feature_map={"Attrition": {'Yes': 1, 'No': 0}},
                                       loss='logistic')
    res['telco_churn'] = calc_original_risk('../datasets/telco_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv',
                                            'Churn',
                                            target_type=str,
                                            feature_map={
                                                'Churn': {'Yes': 1, 'No': -1},
                                            }, loss='logistic'
                                            )
    res['mobile_prices'] = calc_original_risk('../datasets/mobile_prices/train.csv',
                                              'price_range',
                                              target_type=int, )
    res['GenderRecognition'] = calc_original_risk('../datasets/GenderRecognition/voice.csv',
                                                  "label", target_type=str,
                                                  feature_map={"label": {'male': 1, 'female': -1}}, loss='logistic', )
    res['breast'] = calc_risk_load(load_breast_cancer,
                                   loss='logistic', )
    res['diabetes'] = calc_risk_load(load_diabetes, )
    res['iris'] = calc_risk_load(load_iris, feature_map={'target': {0: -1, 1: 1, 2: -1}},
                                 loss='logistic')
    res['load_wine'] = calc_risk_load(load_wine, feature_map={'target': {0: -1, 1: 1, 2: -1}}, loss='logistic')
    print(res)
