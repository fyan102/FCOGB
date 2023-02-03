import numpy as np

from rulefit import RuleFit, Rule
from matplotlib.patches import Rectangle

from realkd.logic import Conjunction, KeyValueProposition, constraint_from_op_string

operator_bound = {
    '<=': 1,  # upper bound
    '<': 1,
    '>=': 0,
    '>': 0
}

operator_comp = {
    '<=': min,  # upper bound
    '<': min,
    '>=': max,
    '>': max
}


def patch_from_rule(rule, fc='blue', axes={'x1': 0, 'x2': 1}, x_min=-4, x_max=4, y_min=-4, y_max=4):
    bounds = [[x_min, x_max], [y_min, y_max]]

    for c in rule.conditions:
        axis = axes[c.feature_name]
        aggr = operator_comp[c.operator]
        bnd = operator_bound[c.operator]
        bounds[axis][bnd] = aggr(c.threshold, bounds[axis][bnd])

    (x, y) = bounds[0][0], bounds[1][0]
    dx = bounds[0][1] - bounds[0][0]
    dy = bounds[1][1] - bounds[1][0]

    return Rectangle((x, y), dx, dy, fill=True, color='black', lw=1, ls='-', fc=fc, alpha=0.2)


def rf_rule_as_query(self):
    props = []
    for cond in self.conditions:
        constraint = constraint_from_op_string(cond.operator, cond.threshold)
        props.append(KeyValueProposition(cond.feature_name, constraint))
    return Conjunction(props)


Rule.as_realkd_query = rf_rule_as_query


def rf_predict_proba(self, X):
    """Predict outcome probability for X, if model type supports probability prediction method
    """

    if 'predict_proba' not in dir(self.lscv):

        error_message = '''
        Probability prediction using predict_proba not available for
        model type {lscv}
        '''.format(lscv=self.lscv)
        raise ValueError(error_message)

    X_concat=np.zeros([X.shape[0],0])
    if 'l' in self.model_type:
        if self.lin_standardise:
            X_concat = np.concatenate((X_concat,self.friedscale.scale(X)), axis=1)
        else:
            X_concat = np.concatenate((X_concat,X), axis=1)
    if 'r' in self.model_type:
        rule_coefs=self.coef_[-len(self.rule_ensemble.rules):]
        if len(rule_coefs)>0:
            X_rules = self.rule_ensemble.transform(X,coefs=rule_coefs)
            if X_rules.shape[0] >0:
                X_concat = np.concatenate((X_concat, X_rules), axis=1)
    return self.lscv.predict_proba(X_concat)


RuleFit.predict_proba = rf_predict_proba


