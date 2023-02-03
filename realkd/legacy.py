"""
Contains early implementation of search objective functions
and bounds mainly for speed comparison and old tests.
"""

import pandas as pd

from math import inf

from realkd.logic import Conjunction, Constraint, KeyValueProposition
from realkd.search import Context


def cov_squared_dev(labels):
    n = len(labels)
    global_mean = sum(labels) / n

    def f(count, mean):
        return count/n * pow(mean - global_mean, 2)

    return f


def impact_count_mean(labels):
    """
    >>> labels = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    >>> f = impact_count_mean(labels)
    >>> f(5, 0.4) # 1/2 * (2/5-1/5) = 1/6
    0.1
    """
    n = len(labels)
    m0 = sum(labels)/n

    def f(c, m):
        return c/n * (m - m0)

    return f


class DfWrapper:

    def __init__(self, df): self.df = df

    def __getitem__(self, item): return self.df.iloc[item]

    def __len__(self): return len(self.df)

    def __iter__(self):
        return (r for (_, r) in self.df.iterrows())


class Impact:
    """
    Impact objective function for conjunctive queries with respect to a specific
    dataset D and target variable y. Formally:

    impact(q) = |ext(q)|/|D| (mean(y; ext(q)) - mean(y; D)) .

    Accepts list-like, dict-like, and Pandas dataframe objects. For example:
    >>> titanic = pd.read_csv("../datasets/titanic/train.csv")
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    >>> old_male = Conjunction([KeyValueProposition('Age', Constraint.greater_equals(60)),
    ...                         KeyValueProposition('Sex', Constraint.equals('male'))])
    >>> imp_survival = Impact(titanic, 'Survived')
    >>> imp_survival(old_male)
    -0.006110487591969073
    >>> imp_survival.exhaustive(verbose=True)
    <BLANKLINE>
    Found optimum after inspecting 92 nodes
    Sex==female
    """

    def _mean(self, q):
        s, c = 0.0, 0.0
        for r in filter(q, self.data):
            s += r[self.target]
            c += 1
        return s/c

    def _coverage(self, q):
        return sum(1 for _ in filter(q, self.data))/self.m

    def __init__(self, data, target):
        self.m = len(data)
        self.data = DfWrapper(data) if isinstance(data, pd.DataFrame) else data
        self.target = target
        self.average = self._mean(lambda _: True)

    def __call__(self, q):
        return self._coverage(q) * (self._mean(q) - self.average)

    def search(self, verbose=False):
        ctx = Context.from_df(self.data.df, without=[self.target], max_col_attr=10)
        f = impact(self.data.df[self.target])
        g = cov_incr_mean_bound(self.data.df[self.target], impact_count_mean(self.data.df[self.target]))
        return ctx.exhaustive(f, g, verbose=verbose)


class SquaredLossObjective:
    """
    Rule boosting objective function for squared loss.

    >>> titanic = pd.read_csv("../datasets/titanic/train.csv")
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    >>> obj = SquaredLossObjective(titanic, titanic['Survived'])
    >>> female = Conjunction([KeyValueProposition('Sex', Constraint.equals('female'))])
    >>> first_class = Conjunction([KeyValueProposition('Pclass', Constraint.less_equals(1))])
    >>> obj(female)
    0.19404590848327577
    >>> reg_obj = SquaredLossObjective(titanic.drop(columns=['Survived']), titanic['Survived'], reg=2)
    >>> reg_obj(female)
    0.19342988972618597
    >>> reg_obj(first_class)
    0.09566220318908493
    >>> reg_obj._mean(female)
    0.7420382165605095
    >>> reg_obj._mean(first_class)
    0.6296296296296297
    >>> reg_obj.exhaustive()
    Sex==female
    """

    def __init__(self, data, target, reg=0):
        """
        :param data:
        :param target: _series_ of target values of matching dimension
        :param reg:
        """
        self.m = len(data)
        self.data = DfWrapper(data) if isinstance(data, pd.DataFrame) else data
        self.target = target
        self.reg = reg

    def _f(self, count, mean):
        return self._reg_term(count)*count/self.m * pow(mean, 2)

    def _reg_term(self, c):
        return 1 / (1 + self.reg / (2 * c))

    def _count(self, q): #almost code duplication: Impact
        return sum(1 for _ in filter(q, self.data))

    def _mean(self, q): #code duplication: Impact
        s, c = 0.0, 0.0
        for i in range(self.m):
            if q(self.data[i]):
                s += self.target[i]
                c += 1
        return s/c

    def search(self, max_col_attr=10):
        # here we need the function in list of row indices; can we save some of these conversions?
        def f(rows):
            c = len(rows)
            if c == 0:
                return 0.0
            m = sum(self.target[i] for i in rows) / c
            return self._f(c, m)

        g = cov_mean_bound(self.target, lambda c, m: self._f(c, m))

        ctx = Context.from_df(self.data.df, max_col_attr=max_col_attr)
        return ctx.exhaustive(f, g)

    def opt_value(self, rows):
        s, c = 0.0, 0
        for i in rows:
            s += self.target[i]
            c += 1

        return s / (self.reg/2 + c) if (c > 0 or self.reg > 0) else 0.0

    def __call__(self, q):
        c = self._count(q)
        m = self._mean(q)
        return self._f(c, m)


def impact(labels):
    """
    Compiles objective function for extension I defined by
    f(I) = len(I)/n (mean_I(l)-mean(l)) for some set of labels l of size n.

    >>> labels = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    >>> f = impact(labels)
    >>> f([0, 1, 2, 3, 4]) # 0.5 * (0.4 - 0.2)
    0.1
    >>> f(range(len(labels)))
    0.0
    """
    g = impact_count_mean(labels)

    def f(extension):
        if len(extension) == 0:
            return -inf
        m = sum((labels[i] for i in extension))/len(extension)
        return g(len(extension), m)

    return f


def squared_loss_obj(labels):
    """
    Builds objective function that maps index set to product
    of relative size of index set times squared difference
    of mean label value described by index set to overall
    mean label value. For instance:

    >>> labels = [-4, -2, -1, -1, 0, 1, 10, 21]
    >>> sum(labels)/len(labels)
    3.0
    >>> obj = squared_loss_obj(labels)
    >>> obj([4, 5, 6, 7])  # local avg 8, relative size 1/2
    12.5

    :param labels: y-values
    :return: f(I) = |I|/n * (mean(y)-mean_I(y))^2
    """

    f = cov_squared_dev(labels)

    def label(i): return labels[i]

    def obj(extent):
        k = len(extent)
        local_mean = sum(map(label, extent)) / k

        return f(k, local_mean)

    return obj


def cov_incr_mean_bound(labels, f):
    """
    >>> labels = [1, 1, 1, 0, 1, 0, 1, 0, 0, 0]
    >>> f = impact_count_mean(labels)
    >>> g = cov_incr_mean_bound(labels, f)
    >>> g(range(len(labels)))
    0.25
    """

    def label(i): return labels[i]

    def bound(extent):
        ordered = sorted(extent, key=label)
        k = len(ordered)
        opt = -inf

        s = 0
        for i in range(k):
            s += labels[ordered[-i-1]]
            opt = max(opt, f(i+1, s/(i+1)))

        return opt

    return bound


def cov_mean_bound(labels, f):
    """
    >>> labels = [-13, -2, -1, -1, 0, 1, 19, 21]
    >>> f = cov_squared_dev(labels)
    >>> obj = squared_loss_obj(labels)
    >>> obj(range(6,8))
    72.25
    >>> f(2, 20.0)
    72.25
    >>> bound = cov_mean_bound(labels, f)
    >>> bound(range(len(labels)))  # local avg 8, relative size 1/2
    72.25

    :param labels:
    :param f: any function that can be re-written as the maximum f(c, m)=max(g(c,m), h(c,m)) over functions g and h
              where g is monotonically increasing in its first and second argument (count and mean)
              and h is monotonically increasing in its first argument and monotonically decreasing in its second
              argument
    :return: bounding function that returns for any set of indices I, the maximum f-value over subsets J <= I
             where f is evaluated as f(|J|, mean(labels; J))
    """

    def label(i): return labels[i]

    def bound(extent):
        ordered = sorted(extent, key=label)
        k = len(ordered)
        opt = -inf

        s = 0
        for i in range(k):
            s += labels[ordered[-i-1]]
            opt = max(opt, f(i+1, s/(i+1)))

        s = 0
        for i in range(k):
            s += labels[ordered[i]]
            opt = max(opt, f(i+1, s/(i+1)))

        return opt

    return bound

