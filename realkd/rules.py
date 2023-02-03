"""
Loss functions and models for rule learning.
"""

import collections.abc

from math import inf, sqrt

import numpy as np
import scipy
from numpy import arange, argsort, array, cumsum, exp, full_like, log2, stack, zeros, zeros_like, log
from pandas import qcut, Series
from sklearn.base import BaseEstimator, clone
# from math import log
from realkd.search import Conjunction, Context, KeyValueProposition, Constraint


class SquaredLoss:
    """
    Squared loss function l(y, s) = (y-s)^2.

    >>> squared_loss
    squared_loss
    >>> y = array([-2, 0, 3])
    >>> s = array([0, 1, 2])
    >>> squared_loss(y, s)
    array([4, 1, 1])
    >>> squared_loss.g(y, s)
    array([ 4,  2, -2])
    >>> squared_loss.h(y, s)
    array([2, 2, 2])
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SquaredLoss, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def __call__(y, s):
        return (y - s) ** 2

    @staticmethod
    def predictions(s):
        return s

    @staticmethod
    def g(y, s):
        return -2 * (y - s)

    @staticmethod
    def h(y, s):
        return full_like(s, 2)

    @staticmethod
    def __repr__():
        return 'squared_loss'

    @staticmethod
    def __str__():
        return 'squared'

    @staticmethod
    def pw(y, s, q):
        return 2 * q * (s - y)


class LogisticLoss:
    """
    Logistic loss function l(y, s) = log2(1 + exp(-ys)).

    Function assumes that positive and negative values are encoded as +1 and -1, respectively.

    >>> y = array([1, -1, 1, -1])
    >>> s = array([0, 0, 10, 10])
    >>> logistic_loss(y, s)
    array([1.00000000e+00, 1.00000000e+00, 6.54967668e-05, 1.44270159e+01])
    >>> logistic_loss.g(y, s)
    array([-5.00000000e-01,  5.00000000e-01, -4.53978687e-05,  9.99954602e-01])
    >>> logistic_loss.h(y, s)
    array([2.50000000e-01, 2.50000000e-01, 4.53958077e-05, 4.53958077e-05])
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogisticLoss, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def __call__(y, s):
        return log2(1 + exp(-y * s))

    @staticmethod
    def sigmoid(a):
        return 1 / (1 + exp(-a))

    @staticmethod
    def predictions(s):
        preds = zeros_like(s)
        preds[s >= 0] = 1
        preds[s < 0] = -1
        return preds  # this case now returns np array

    @staticmethod
    def probabilities(s):
        pos = LogisticLoss.sigmoid(s)
        return stack((1 - pos, pos), axis=1)

    @staticmethod
    def g(y, s):
        return -y * LogisticLoss.sigmoid(-y * s)

    @staticmethod
    def h(y, s):
        sig = LogisticLoss.sigmoid(-y * s)
        return sig * (1.0 - sig)

    @staticmethod
    def __repr__():
        return 'logistic_loss'

    @staticmethod
    def __str__():
        return 'logistic'

    @staticmethod
    def pw(y, s, q):
        return -y * q * exp(-y * s) / (1 + exp(-y * s)) / log(2)


class PoissonLoss:
    """
    Poisson Loss function l(y, s) = exp(s) - s * y +log(y) * y - y

    s is the log value of the actual predicted value
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PoissonLoss, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def __call__(y, s):
        res = zeros_like(y) * 1.0
        y = np.array(y, dtype=np.float64)
        loss = exp(s) - s * y + y * log(y, where=y > 0, out=res) - y
        return loss

    @staticmethod
    def predictions(s):
        return exp(s)

    @staticmethod
    def g(y, s):
        res = exp(s) - y
        return res

    @staticmethod
    def h(y, s):
        res = exp(s)
        return res

    @staticmethod
    def __repr__():
        return 'poisson_loss'

    @staticmethod
    def __str__():
        return 'poisson'

    @staticmethod
    def pw(y, s, q):
        return q * (exp(s) - y)


logistic_loss = LogisticLoss()
squared_loss = SquaredLoss()
poisson_loss = PoissonLoss()

#: Dictionary of available loss functions with keys corresponding to their string representations.
loss_functions = {
    LogisticLoss.__repr__(): logistic_loss,
    SquaredLoss.__repr__(): squared_loss,
    LogisticLoss.__str__(): logistic_loss,
    SquaredLoss.__str__(): squared_loss,
    PoissonLoss.__repr__(): poisson_loss,
    PoissonLoss.__str__(): poisson_loss
}


def loss_function(loss):
    """Provides loss functions from string representation.

    :param loss: string identifier of loss function loss function
    :return: loss function matching corresponding to input string (or unchanged input if was already loss function)
    """
    if callable(loss):
        return loss
    else:
        return loss_functions[loss]


class Rule:
    """
    Represents a rule of the form "r(x) = y if q(x) else z"
    for some binary query function q.

    >>> import pandas as pd
    >>> titanic = pd.read_csv('../datasets/titanic/train.csv')
    >>> titanic[['Name', 'Sex', 'Survived']].iloc[0]
    Name        Braund, Mr. Owen Harris
    Sex                            male
    Survived                          0
    Name: 0, dtype: object
    >>> titanic[['Name', 'Sex', 'Survived']].iloc[1]
    Name        Cumings, Mrs. John Bradley (Florence Briggs Th...
    Sex                                                    female
    Survived                                                    1
    Name: 1, dtype: object

    >>> female = KeyValueProposition('Sex', Constraint.equals('female'))
    >>> r = Rule(female, 1.0, 0.0)
    >>> r(titanic.iloc[0]), r(titanic.iloc[1])
    (0.0, 1.0)

    >>> empty = Rule()
    >>> empty
       +0.0000 if True
    """

    def __init__(self, q=Conjunction([]), y=0.0, z=0.0):
        """
        :param `~realkd.logic.Conjunction` q: rule query (antecedent/condition)
        :param float y: prediction value if query satisfied
        :param float z: prediction value if query not satisfied
        """
        self.q = q
        self.y = y
        self.z = z

    def __call__(self, x):
        """ Predicts score for input data based on loss function.

        For instance for logistic loss will return log odds of the positive class.

        :param ~pandas.DataFrame x: input data
        :return: :class:`~numpy.array` of prediction scores (one for each rows in x)
        """
        sat = self.q(x)
        return sat * self.y + (1 - sat) * self.z

    def __repr__(self):
        # TODO: if existing also print else part
        return f'{self.y:+10.4f} if {self.q}'


class AdditiveRuleEnsemble:
    """Rules ensemble that combines scores of its member rules additively to form predictions.

    While order of rules does not influence predictions, it is important for indexing and
    slicing, which provides convenient access to individual ensemble members and modified
    ensembles.

    For example:

    >>> female = KeyValueProposition('Sex', Constraint.equals('female'))
    >>> r1 = Rule(Conjunction([]), -0.5, 0.0)
    >>> r2 = Rule(female, 1.0, 0.0)
    >>> r3 = Rule(female, 0.3, 0.0)
    >>> r4 = Rule(Conjunction([]), -0.2, 0.0)
    >>> ensemble = AdditiveRuleEnsemble(members=[r1, r2, r3, r4])
    >>> len(ensemble)
    4
    >>> ensemble[2]
       +0.3000 if Sex==female
    >>> ensemble[:2]
       -0.5000 if True
       +1.0000 if Sex==female
    """

    def __init__(self, members=[]):
        """

        :param List[Rule] members: the individual rules that make up the ensemble
        """
        self.members = members[:]

    def __repr__(self):
        return str.join('\n', (str(r) for r in self.members))

    def __len__(self):
        """Length of the ensemble.

        :return: number of contained rules
        """
        return len(self.members)

    def __getitem__(self, item):
        """Index access to the individual members of the ensemble.

        Also supports slicing, resulting in a new ensemble.

        :param int item: index
        :return: rule of index
        """
        if isinstance(item, slice):
            _members = self.members[item]
            return AdditiveRuleEnsemble(_members)
        else:
            return self.members[item]

    def __call__(self, x):  # look into swapping to Series and numpy
        """Computes combined prediction scores using all ensemble members.

        :param ~pandas.DataFrame x: input data
        :return: :class:`~numpy.array` of prediction scores (one for each rows in x)
        """
        res = zeros(len(x))  # TODO: a simple reduce should do if we can rule out empty ensemble
        for r in self.members:
            res += r(x)
        return res

    def append(self, rule):
        """Adds a rule to the ensemble.

        :param Rule rule: the rule to be added
        :return: self
        """
        self.members.append(rule)
        return self

    def size(self):
        """ Computes the total size of the ensemble.

        Currently, this is defined as the number of rules (length of the ensemble)
        plus the the number of elementary conditions in all rule queries.

        In the future this is subject to change to a more general notion of size (taking into account
        the possible greater number of parameters of more complex rules).

        :return: size of ensemble as defined above
        """
        return sum(len(r.q) for r in self.members) + len(self.members)

    def consolidated(self, inplace=False):
        """ Consolidates rules with equivalent queries into one.

        :param bool inplace: whether to update self or to create new ensemble
        :return: reference to consolidated ensemble (self if inplace=True)

        For example:

        >>> female = KeyValueProposition('Sex', Constraint.equals('female'))
        >>> r1 = Rule(Conjunction([]), -0.5, 0.0)
        >>> r2 = Rule(female, 1.0, 0.0)
        >>> r3 = Rule(female, 0.3, 0.0)
        >>> r4 = Rule(Conjunction([]), -0.2, 0.0)
        >>> ensemble = AdditiveRuleEnsemble([r1, r2, r3, r4])
        >>> ensemble.consolidated(inplace=True) # doctest: +NORMALIZE_WHITESPACE
        -0.7000 if True
        +1.3000 if Sex==female
        """
        _members = self.members[:]
        for i, r1 in enumerate(_members):
            q = r1.q
            y = r1.y
            z = r1.z
            for j in range(len(_members) - 1, i, -1):
                r2 = _members[j]
                if q == r2.q:
                    y += r2.y
                    z += r2.z
                    _members.pop(j)
            _members[i] = Rule(q, y, z)

        if inplace:
            self.members = _members
            return self
        else:
            return AdditiveRuleEnsemble(_members)


class GradientBoostingObjective:
    """
    >>> import pandas as pd
    >>> titanic = pd.read_csv("../datasets/titanic/train.csv")
    >>> survived = titanic['Survived']
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
    >>> obj = GradientBoostingObjective(titanic, survived, reg=0.0)
    >>> female = Conjunction([KeyValueProposition('Sex', Constraint.equals('female'))])
    >>> first_class = Conjunction([KeyValueProposition('Pclass', Constraint.less_equals(1))])
    >>> obj(obj.data[female].index)
    0.1940459084832758
    >>> obj(obj.data[first_class].index)
    0.09610508375940474
    >>> obj.bound(obj.data[first_class].index)
    0.1526374859708193
    >>> reg_obj = GradientBoostingObjective(titanic, survived, reg=2)
    >>> reg_obj(reg_obj.data[female].index)
    0.19342988972618602
    >>> reg_obj(reg_obj.data[first_class].index)
    0.09566220318908492

    >>> q = reg_obj.search(method='exhaustive', verbose=True)
    <BLANKLINE>
    Found optimum after inspecting 103 nodes: [16]
    Greedy simplification: [16]
    >>> q
    Sex==female
    >>> reg_obj.opt_weight(q)
    0.7396825396825397

    >>> obj = GradientBoostingObjective(titanic, survived.replace(0, -1), loss='logistic')
    >>> obj(obj.data[female].index)
    0.04077109318199465
    >>> obj.opt_weight(female)
    0.9559748427672956
    >>> best = obj.search(method='exhaustive', order='bestvaluefirst', verbose=True)
    <BLANKLINE>
    Found optimum after inspecting 446 nodes: [27, 29]
    Greedy simplification: [27, 29]
    >>> best
    Pclass>=2 & Sex==male
    >>> obj(obj.data[best].index)
    0.13072995752734315
    >>> obj.opt_weight(best)
    -1.4248366013071896
    """

    def __init__(self, data, target, predictions=None, loss=SquaredLoss, reg=1.0, rules=None, **kwargs):
        self.loss = loss_function(loss)
        self.reg = reg
        predictions = zeros_like(target) if predictions is None else predictions
        g = array(self.loss.g(target, predictions))
        h = array(self.loss.h(target, predictions))
        r = g / h
        order = argsort(r)[::-1]
        self.g = g[order]
        self.h = h[order]
        self.data = data.iloc[order].reset_index(drop=True)
        self.target = target.iloc[order].reset_index(drop=True)
        self.n = len(target)

    def __call__(self, ext):
        if len(ext) == 0:
            return -inf
        g_q = self.g[ext]
        h_q = self.h[ext]
        return g_q.sum() ** 2 / (2 * self.n * (self.reg + h_q.sum()))

    def bound(self, ext):
        m = len(ext)
        if m == 0:
            return -inf

        g_q = self.g[ext]
        h_q = self.h[ext]

        num_pre = cumsum(g_q) ** 2
        num_suf = cumsum(g_q[::-1]) ** 2
        den_pre = cumsum(h_q) + self.reg
        den_suf = cumsum(h_q[::-1]) + self.reg
        neg_bound = (num_suf / den_suf).max() / (2 * self.n)
        pos_bound = (num_pre / den_pre).max() / (2 * self.n)
        return max(neg_bound, pos_bound)

    def opt_weight(self, q):
        # TODO: this should probably just be defined for ext (saving the q evaluation)
        # ext = self.ext(q)
        ext = self.data.loc[q].index
        g_q = self.g[ext]
        h_q = self.h[ext]
        return -g_q.sum() / (self.reg + h_q.sum())

    def search(self, method='greedy', verbose=False, **search_params):
        from realkd.search import search_methods
        ctx = Context.from_df(self.data, **search_params)
        if verbose >= 2:
            print(f'Created search context with {len(ctx.attributes)} attributes')
        # return getattr(ctx, method)(self, self.bound, verbose=verbose, **search_params)
        return search_methods[method](ctx, self, self.bound, verbose=verbose, **search_params).run()


class XGBRuleEstimator(BaseEstimator):
    r"""
    Fits a rule based on first and second loss derivatives of some prior prediction values.

    In more detail, given some prior prediction values :math:`f(x)` and a twice differentiable loss function
    :math:`l(y,f(x))`, a rule :math:`r(x)=wq(x)` is fitted by finding a binary query :math:`q` via maximizing the objective function

    .. math::

        \mathrm{obj}(q) = \frac{\left( \sum_{i \in I(q)} g_i \right )^2}{2n \left(\lambda + \sum_{i \in I(q)} h_i \right)}


    and finding the optimal weight as

    .. math::

        w = -\frac{\sum_{i \in I(q)} g_i}{\lambda + \sum_{i \in I(q)} h_i} \enspace .

    Here, :math:`I(q)` denotes the indices of training examples selected by :math:`q` and

    .. math::

        g_i=\frac{\mathrm{d} l(y_i, y)}{\mathrm{d}y}\Bigr|_{\substack{y=f(x_i)}} \enspace ,
        \quad
        h_i=\frac{\mathrm{d}^2 l(y_i, y)}{\mathrm{d}y^2}\Bigr|_{\substack{y=f(x_i)}}

    refer to the first and second order gradient statistics of the prior prediction values.


    >>> import pandas as pd
    >>> titanic = pd.read_csv('../datasets/titanic/train.csv')
    >>> target = titanic.Survived
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
    >>> opt = XGBRuleEstimator(reg=0.0)
    >>> opt.fit(titanic, target).rule_
       +0.7420 if Sex==female

    >>> best_logistic = XGBRuleEstimator(loss='logistic')
    >>> best_logistic.fit(titanic, target.replace(0, -1)).rule_
       -1.4248 if Pclass>=2 & Sex==male

    >>> best_logistic.predict(titanic) # doctest: +ELLIPSIS
    array([-1.,  1.,  1.,  1., ...,  1.,  1., -1.])

    >>> greedy = XGBRuleEstimator(loss='logistic', reg=1.0, search='greedy')
    >>> greedy.fit(titanic, target.replace(0, -1)).rule_
       -1.4248 if Pclass>=2 & Sex==male
    """

    # max_col attribute to change number of propositions
    def __init__(self, loss='squared', reg=1.0, search='exhaustive', object_func=GradientBoostingObjective,
                 search_params={'order': 'bestboundfirst', 'apx': 1.0, 'max_depth': None, 'discretization': qcut,
                                'max_col_attr': 10},
                 query=None):
        """
        :param str|callable loss: loss function either specified via string identifier (e.g., ``'squared'`` for regression or ``'logistic'`` for classification) or directly has callable loss function with defined first and second derivative (see :data:`~realkd.rules.loss_functions`)
        :param float reg: the regularization parameter :math:`\\lambda`
        :param str|type search: search method either specified via string identifier (e.g., ``'greedy'`` or ``'exhaustive'``) or directly as search type (see :func:`realkd.search.search_methods`)
        :param dict search_params: parameters to apply to discretization (when creating binary search context from
                              dataframe via :func:`~realkd.search.Context.from_df`) as well as to actual search method
                              (specified by ``method``). See :mod:`~realkd.search`.
        """
        self.reg = reg
        self.loss = loss
        self.search = search
        self.search_params = search_params
        self.query = query
        self.object_func = object_func
        self.rule_ = None

    def decision_function(self, x):
        """ Predicts score for input data based on loss function.

        For instance for logistic loss will return log odds of the positive class.

        :param ~pandas.DataFrame x: input data
        :return: :class:`~numpy.array` of prediction scores (one for each rows in x)

        """
        return self.rule_(x)

    def __repr__(self):
        return f'{type(self).__name__}(reg={self.reg}, loss={self.loss})'

    def fit(self, data, target, scores=None, verbose=False):
        """
        Fits rule to provide best loss reduction on given data
        (where the baseline prediction scores are either given
        explicitly through the scores parameter or are assumed
        to be 0.

        :param data: pandas DataFrame containing only the feature columns
        :param target: pandas Series containing the target values
        :param scores: prior prediction scores according to which the reduction in prediction loss is optimised
        :param verbose: whether to print status update and summary of query search
        :return: self

        """
        obj = self.object_func(data, target, predictions=scores, loss=self.loss, reg=self.reg)
        q = obj.search(method=self.search, verbose=verbose, **self.search_params) if self.query is None else self.query
        y = obj.opt_weight(q)
        self.rule_ = Rule(q, y)
        return self

    def predict(self, data):
        """Generates predictions for input data.

        :param data: pandas dataframe with co-variates for which to make predictions
        :return: array of predictions
        """
        loss = loss_function(self.loss)
        return loss.predictions(self.rule_(data))

    def predict_proba(self, data):
        """Generates probability predictions for input data.

        This method is only supported for suitable loss functions.

        :param data: pandas dataframe with data to predict probabilities for
        :return: array of probabilities (shape according to number of classes)
        """
        loss = loss_function(self.loss)
        return loss.probabilities(self.rule_(data))


class CorrectiveRuleBoostingEstimator(BaseEstimator):
    r"""
    The corrective rule boosting estimator.

    When a new rule has been added to the rule ensemble, the weight of each rule is re-calculated
    to get a smaller risk value.

    >>> import datasets
    >>> x, y = datasets.alternating_block_model(12, 5, 5)
    >>> res = CorrectiveRuleBoostingEstimator(num_rules = 5,reg = 0,search_params = {'max_col_attr': 200}).fit(x, y)
    >>> res.rules_
    +1.0000 if
    -2.0000 if x<=33 & x>=29
    -2.0000 if x<=50 & x>=46
    -2.0000 if x<=16 & x>=12
    -2.0000 if x<=67 & x>=63

    """

    def __init__(self, num_rules=3, reg=1.0, verbose=False, loss='squared', search='greedy',
                 search_params={'order': 'bestboundfirst', 'apx': 1.0, 'max_depth': None, 'discretization': qcut,
                                'max_col_attr': 10}, rules=None, obj_func=GradientBoostingObjective,
                 search_method='Newton-CG'):
        """
        :param int num_rules: the number of rules
        :param float reg: the regularization parameter :math:`\\lambda`
        :param dict search_params: parameters for XGBRuleEstimator to do the searching.
                                   See :mod:`~realkd.rules.XGBRuleEstimator`.
        """
        self.num_rules = num_rules
        self.verbose = verbose
        self.reg = reg
        self.loss = loss
        self.search_params = search_params
        self.search = search
        if rules is None:
            self.rules_ = AdditiveRuleEnsemble([])
        else:
            self.rules_ = rules
        self.history = []
        self.obj_func = obj_func
        self.search_method = search_method

    def fit_squared(self, data, target):
        """
        Fit rules using the input data and target. This method will recalculate the weight of each
        rule in each iteration. This method only works for squared loss.

        :param data: the pandas DataFrame containing the attributes of each data point
        :param target: the pandas Series containing the labels of each data point
        :return: self
        """
        while len(self.rules_) < self.num_rules:
            scores = self.rules_(data)
            estimator = XGBRuleEstimator(loss='squared', reg=self.reg, search_params=self.search_params)
            estimator.fit(data, target, scores, max(self.verbose - 1, 0))
            if self.verbose:
                print(estimator.rule_)
            self.rules_.append(estimator.rule_)
            try:
                q_mat = np.column_stack([self.rules_[i].q(data) + np.zeros(len(data)) for i in range(len(self.rules_))])
                w = np.linalg.solve(q_mat.T.dot(q_mat), q_mat.T.dot(target))
                for i in range(len(self.rules_)):
                    self.rules_[i].y = w[i]
            except Exception as e:
                pass
        return self

    @staticmethod
    def get_risk(loss, y, q_mat, reg):
        """
        Get the function of calculating sum of loss plus the regularization term.

        :param loss: the loss function
        :param y: the target values
        :param q_mat: the n * k matrix which contains the query values of each rule on each data point
        :param reg: regularization parameter
        :return: a function which calculates the sum of loss plus the regularization term.
        """

        def sum_loss(weights):
            return sum(loss(y, q_mat.dot(weights))) + reg * sum(weights * weights) / 2

        return sum_loss

    @staticmethod
    def get_gradient(g, y, q_mat, reg):
        """
        Get the gradient of the sum of loss in terms of weights

        :param g: the derivative of the loss function
        :param y: the target values
        :param q_mat: the n * k matrix which contains the query values of each rule on each data point
        :param reg: regularization parameter
        :return: a function which calculates the gradient of the sum of loss in terms of weights
        """

        def gradient(weights):
            grad_vec = g(y, q_mat.dot(weights))
            return q_mat.T.dot(grad_vec) + reg * weights

        return gradient

    @staticmethod
    def get_hessian(h, y, q_mat, reg):
        """
        Get the Hessian Matrix of sum of loss in terms of weights

        :param h: the secondary derivative of the loss function
        :param y: the target values
        :param q_mat: the n * k matrix which contains the query values of each rule on each data point
        :param reg: regularization parameter
        :return: a function which calculates the Hessian Matrix of the sum of loss in terms of weights
        """

        def hessian(weights):
            h_vec = h(y, q_mat.dot(weights))
            return q_mat.T.dot(np.diag(h_vec)).dot(q_mat) + np.diag([reg] * len(weights))

        return hessian

    def fit(self, data, target):
        """
        Fit rules using the input data and target. This method will recalculate the weight of each
        rule in each iteration.

        :param data: the pandas DataFrame containing the attributes of each data point
        :param target: the pandas Series containing the labels of each data point
        :return: self
        """
        self.rules_.members = []
        self.history = []
        while len(self.rules_) < self.num_rules:
            scores = self.rules_(data)
            estimator = XGBRuleEstimator(loss=self.loss, reg=self.reg, search=self.search,
                                         search_params=self.search_params, object_func=self.obj_func)
            estimator.fit(data, target, scores, max(self.verbose - 1, 0))
            if self.verbose:
                print(estimator.rule_)
            self.rules_.append(estimator.rule_)
            try:
                w = self.fully_correct(data, target, self.rules_)
                for i in range(len(self.rules_)):
                    self.rules_[i].y = w[i]
                self.history.append(AdditiveRuleEnsemble([Rule(q=rule.q, y=rule.y) for rule in self.rules_.members]))
            except Exception as e:
                print('Error 5', e)
                self.history.append(AdditiveRuleEnsemble([Rule(q=rule.q, y=rule.y) for rule in self.rules_.members]))
        return self

    def fully_correct(self, data, target, rules):
        g = loss_function(self.loss).g
        h = loss_function(self.loss).h
        loss = loss_function(self.loss)

        # w = np.array([rules[i].y for i in range(int(len(rules)))])
        y = np.array(target)
        q_mat = np.column_stack([rules[i].q(data) + np.zeros(len(data)) for i in range(len(rules))])
        sum_loss = self.get_risk(loss, y, q_mat, self.reg)
        gradient = self.get_gradient(g, y, q_mat, self.reg)
        hessian = self.get_hessian(h, y, q_mat, self.reg)

        def norm(xs):
            return sqrt(sum([x * x for x in xs]))

        if self.search_method == 'GD':  # Gradient descent
            w = np.array([3.5 if r.y > 100 and self.loss == 'poisson' else r.y for r in rules])
            old_w = zeros_like(w) * 1.0
            i = 0
            while norm(old_w - w) > 1e-3 and i < 20:
                old_w = np.array(w)
                if norm(gradient(w)) == 0:
                    break
                p = -gradient(w) / norm(gradient(w))
                left = 0
                right = norm(w)
                w += self.golden_ratio_search(sum_loss, left, right, p, old_w) * p
                i += 1
        elif self.search_method == 'Line':
            w = np.array([3.5 if r.y > 100 and self.loss == 'poisson' else r.y for r in rules])
            if norm(gradient(w)) != 0:
                p = -gradient(w) / norm(gradient(w))
                left = 0
                right = norm(w)
                distance = self.golden_ratio_search(sum_loss, left, right, p, w)
                w += distance * p
        else:
            w = np.array([rules[i].y for i in range(int(len(rules)))])
            w = scipy.optimize.minimize(sum_loss, w, method=self.search_method, jac=gradient, hess=hessian,
                                        options={'disp': False}).x
        return w

    def golden_ratio_search(self, func, left, right, dir, origin):
        ratio = (sqrt(5) - 1) / 2
        while right - left > 1e-3:
            lam = left + (1 - ratio) * (right - left)
            mu = left + ratio * (right - left)
            r_lam = func(origin + lam * dir)
            r_mu = func(origin + mu * dir)
            if r_lam <= r_mu:
                right = mu
            else:
                left = lam
        return (left + right) / 2

    def predict(self, data):
        return self.rules_(data)


class RuleBoostingEstimator(BaseEstimator):
    """Additive rule ensemble fitted by boosting.

    That is, rules are fitted iteratively by one or more base learners until a desired number of rules has been
    learned. In each iteration, the base learner fits the training data taking into account the prediction scores
    of the already fixed part of the ensemble.

    Therefore, base learners need to provide a fit method that can take into account prior predictions
    (see :func:`XGBRuleEstimator.fit`).

    >>> import pandas as pd
    >>> from sklearn.metrics import roc_auc_score
    >>> titanic = pd.read_csv('../datasets/titanic/train.csv')
    >>> survived = titanic.Survived
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
    >>> re = RuleBoostingEstimator(base_learner=XGBRuleEstimator(loss=logistic_loss))
    >>> re.fit(titanic, survived.replace(0, -1), verbose=0) # doctest: +SKIP
       -1.4248 if Pclass>=2 & Sex==male
       +1.7471 if Pclass<=2 & Sex==female
       +2.5598 if Age<=19.0 & Fare>=7.8542 & Parch>=1.0 & Sex==male & SibSp<=1.0

    Multiple base learners can be specified and are used sequentially. The last based learner is used as many times
    as necessary to learn the desired number of rules. This mechanism can, e.g., be used to fit an "offset rule":

    >>> re_with_offset = RuleBoostingEstimator(num_rules=2, base_learner=[XGBRuleEstimator(loss='logistic', query = Conjunction([])), XGBRuleEstimator(loss='logistic')])
    >>> re_with_offset.fit(titanic, survived.replace(0, -1)).rules_
       -0.4626 if True
       +2.3076 if Pclass<=2 & Sex==female

    >>> greedy = RuleBoostingEstimator(num_rules=3, base_learner=XGBRuleEstimator(loss='logistic', search='greedy'))
    >>> greedy.fit(titanic, survived.replace(0, -1)).rules_ # doctest: -SKIP
       -1.4248 if Pclass>=2 & Sex==male
       +1.7471 if Pclass<=2 & Sex==female
       -0.4225 if Parch<=1.0 & Sex==male
    >>> roc_auc_score(survived, greedy.rules_(titanic))
    0.8321136782454011
    >>> opt = RuleBoostingEstimator(num_rules=3, base_learner=XGBRuleEstimator(loss='logistic', search='exhaustive'))
    >>> opt.fit(titanic, survived.replace(0, -1)).rules_ # doctest: -SKIP
       -1.4248 if Pclass>=2 & Sex==male
       +1.7471 if Pclass<=2 & Sex==female
       +2.5598 if Age<=19.0 & Fare>=7.8542 & Parch>=1.0 & Sex==male & SibSp<=1.0
    >>> roc_auc_score(survived, opt.rules_(titanic)) # doctest: -SKIP
    0.8490530363553084
    """

    def __init__(self, num_rules=3, base_learner=XGBRuleEstimator(loss='squared', reg=1.0, search='greedy'),
                 verbose=False, fully_correction=False, correction_method='Newton-CG'):
        """

        :param int num_rules: the desired number of ensemble members
        :param Estimator|Sequence[Estimator] base_learner: the base learner(s) to be used in each iteration (last base
                                    learner is used as many times as necessary to fit desired number of rules)
        :param bool fully_correction: do a fully-correction step after generating a rule if True
        :correction_method: the method to do the fully-correction

        """
        self.num_rules = num_rules
        self.base_learner = base_learner
        self.rules_ = AdditiveRuleEnsemble([])
        self.verbose = verbose
        self.fully_correction = fully_correction
        self.correction_method = correction_method
        self.history = []
        self.reg = base_learner.reg
        self.loss = base_learner.loss

    def _next_base_learner(self):
        if isinstance(self.base_learner, collections.abc.Sequence):
            return self.base_learner[min(len(self.rules_), len(self.base_learner) - 1)]
        else:
            return clone(self.base_learner)

    def decision_function(self, x):
        """Computes combined prediction scores using all ensemble members.

        :param ~pandas.DataFrame x: input data

        :return: :class:`~numpy.array` of prediction scores (one for each rows in x)
        """
        return self.rules_(x)

    def __repr__(self, N_CHAR_MAX=700):
        return f'{type(self).__name__}(max_rules={self.num_rules}, base_learner={self.base_learner})'

    def fit(self, data, target):
        self.history = []
        while len(self.rules_) < self.num_rules:
            scores = self.rules_(data)
            estimator = self._next_base_learner()
            estimator.fit(data, target, scores, max(self.verbose - 1, 0))
            if self.verbose:
                print(estimator.rule_)
            self.rules_.append(estimator.rule_)
            if self.fully_correction:
                try:
                    w = self.fully_correct(data, target, self.rules_)
                    for i in range(len(self.rules_)):
                        self.rules_[i].y = w[i]
                    self.history.append(
                        AdditiveRuleEnsemble([Rule(q=rule.q, y=rule.y) for rule in self.rules_.members]))
                except Exception as e:
                    print('Error 5', e)
                    self.history.append(
                        AdditiveRuleEnsemble([Rule(q=rule.q, y=rule.y) for rule in self.rules_.members]))
        return self

    def fully_correct(self, data, target, rules):
        g = loss_function(self.loss).g
        h = loss_function(self.loss).h
        loss = loss_function(self.loss)
        y = np.array(target)
        q_mat = np.column_stack([rules[i].q(data) + np.zeros(len(data)) for i in range(len(rules))])
        sum_loss = self.get_risk(loss, y, q_mat, self.reg)
        gradient = self.get_gradient(g, y, q_mat, self.reg)
        hessian = self.get_hessian(h, y, q_mat, self.reg)

        def norm(xs):
            return sqrt(sum([x * x for x in xs]))

        if self.correction_method == 'GD':  # Gradient descent
            w = np.array([3.5 if r.y > 100 and self.loss == 'poisson' else r.y for r in rules])
            old_w = zeros_like(w) * 1.0
            while norm(old_w - w) > 1e-3:
                old_w = np.array(w)
                p = -gradient(w) / norm(gradient(w))
                left = 0
                right = norm(w)
                w += self.golden_ratio_search(sum_loss, left, right, p, old_w) * p
        elif self.correction_method == 'Line':  # Line search
            w = np.array([3.5 if r.y > 100 and self.loss == 'poisson' else r.y for r in rules])
            p = -gradient(w) / norm(gradient(w))
            left = 0
            right = norm(w)
            distance = self.golden_ratio_search(sum_loss, left, right, p, w)
            w += distance * p
        else:
            w = np.array([rules[i].y for i in range(int(len(rules)))])
            w = scipy.optimize.minimize(sum_loss, w, method=self.correction_method, jac=gradient, hess=hessian,
                                        options={'disp': False}).x
        return w

    @staticmethod
    def golden_ratio_search(func, left, right, direction, origin, epsilon=1e-3):
        """
        Use golden ratio search to search for an optimal distance along a direction
        to make the function minimized

        :param func: function to be minimized
        :param left: left bound of the search interval
        :param right: right bound of the search interval
        :param direction: search direction
        :param origin: origin point
        :param epsilon: the precision of the search
        """
        ratio = (sqrt(5) - 1) / 2
        while right - left > epsilon:
            lam = left + (1 - ratio) * (right - left)
            mu = left + ratio * (right - left)
            r_lam = func(origin + lam * direction)
            r_mu = func(origin + mu * direction)
            if r_lam <= r_mu:
                right = mu
            else:
                left = lam
        return (left + right) / 2

    @staticmethod
    def get_risk(loss, y, q_mat, reg):
        """
        Get the function of calculating sum of loss plus the regularization term.

        :param loss: the loss function
        :param y: the target values
        :param q_mat: the n * k matrix which contains the query values of each rule on each data point
        :param reg: regularization parameter
        :return: a function which calculates the sum of loss plus the regularization term.
        """

        def sum_loss(weights):
            return sum(loss(y, q_mat.dot(weights))) + reg * sum(weights * weights) / 2

        return sum_loss

    @staticmethod
    def get_gradient(g, y, q_mat, reg):
        """
        Get the gradient of the sum of loss in terms of weights

        :param g: the derivative of the loss function
        :param y: the target values
        :param q_mat: the n * k matrix which contains the query values of each rule on each data point
        :param reg: regularization parameter
        :return: a function which calculates the gradient of the sum of loss in terms of weights
        """

        def gradient(weights):
            grad_vec = g(y, q_mat.dot(weights))
            return q_mat.T.dot(grad_vec) + reg * weights

        return gradient

    @staticmethod
    def get_hessian(h, y, q_mat, reg):
        """
        Get the Hessian Matrix of sum of loss in terms of weights

        :param h: the secondary derivative of the loss function
        :param y: the target values
        :param q_mat: the n * k matrix which contains the query values of each rule on each data point
        :param reg: regularization parameter
        :return: a function which calculates the Hessian Matrix of the sum of loss in terms of weights
        """

        def hessian(weights):
            h_vec = h(y, q_mat.dot(weights))
            return q_mat.T.dot(np.diag(h_vec)).dot(q_mat) + np.diag([reg] * len(weights))

        return hessian

    def predict(self, data):
        loss = loss_function(self._next_base_learner().loss)
        return loss.predictions(self.rules_(data))

    def predict_proba(self, data):
        loss = loss_function(self._next_base_learner().loss)
        return loss.probabilities(self.rules_(data))


if __name__ == '__main__':
    import doctest

    doctest.testmod()
