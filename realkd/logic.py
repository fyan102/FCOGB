"""
Elements of propositional logic: constraints, propositions, and
conjunctions.
"""

import pandas as pd
import re

from numpy import logical_and, ones


class Constraint:
    """
    Boolean condition on a single value with string representation. For example:
    >>> t = 21
    >>> c = Constraint.less_equals(21)
    >>> c
    Constraint(x<=21)
    >>> format(c, 'age')
    'age<=21'
    >>> c(18)
    True
    >>> c(63)
    False
    >>> import numpy as np
    >>> a =  np.arange(15, 25)
    >>> c(a)
    array([ True,  True,  True,  True,  True,  True,  True, False, False,
           False])
    """

    def __init__(self, cond, str_repr=None):
        self.cond = cond
        self.str_repr = str_repr or (lambda vn: str(cond)+'('+vn+')')

    def __call__(self, value):
        return self.cond(value)

    def __format__(self, varname):
        return self.str_repr(varname)

    def __repr__(self):
        return 'Constraint('+format(self, 'x')+')'

    @staticmethod
    def less_equals(value):
        return Constraint(lambda v: v <= value, lambda n: str(n)+'<='+str(value))

    @staticmethod
    def less(value):
        return Constraint(lambda v: v < value, lambda n: str(n)+'<'+str(value))

    @staticmethod
    def greater_equals(value):
        return Constraint(lambda v: v >= value, lambda n: str(n)+'>='+str(value))

    @staticmethod
    def greater(value):
        return Constraint(lambda v: v > value, lambda n: str(n)+'>'+str(value))

    @staticmethod
    def equals(value):
        return Constraint(lambda v: v == value, lambda n: str(n)+'=='+str(value))

    @staticmethod
    def not_equals(value):
        return Constraint(lambda v: v != value, lambda n: str(n)+'!='+str(value))


_operator_factory = {
    '==': Constraint.equals,
    '!=': Constraint.not_equals,
    '>': Constraint.greater,
    '<': Constraint.less,
    '>=': Constraint.greater_equals,
    '<=': Constraint.less_equals
}


def constraint_from_op_string(op, value):
    return _operator_factory[op](value)


class KeyValueProposition:
    """
    Callable proposition that represents constraint on value for some fixed key in a dict-like object
    such as Pandas row series.

    For example:

    >>> titanic = pd.read_csv("../datasets/titanic/train.csv")
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    >>> male = KeyValueProposition('Sex', Constraint.equals('male'))
    >>> male
    Sex==male

    ---> WARNING: string values need probably be quoted in representation to work as pandas query as intended

    >>> titanic.iloc[10]
    Survived         1
    Pclass           3
    Sex         female
    Age            4.0
    SibSp            1
    Parch            1
    Fare          16.7
    Embarked         S
    Name: 10, dtype: object

    >>> male(titanic.iloc[10])
    False
    >>> titanic.loc[male]
         Survived  Pclass   Sex   Age  SibSp  Parch     Fare Embarked
    0           0       3  male  22.0      1      0   7.2500        S
    4           0       3  male  35.0      0      0   8.0500        S
    5           0       3  male   NaN      0      0   8.4583        Q
    6           0       1  male  54.0      0      0  51.8625        S
    7           0       3  male   2.0      3      1  21.0750        S
    ..        ...     ...   ...   ...    ...    ...      ...      ...
    883         0       2  male  28.0      0      0  10.5000        S
    884         0       3  male  25.0      0      0   7.0500        S
    886         0       2  male  27.0      0      0  13.0000        S
    889         1       1  male  26.0      0      0  30.0000        C
    890         0       3  male  32.0      0      0   7.7500        Q
    <BLANKLINE>
    [577 rows x 8 columns]

    >>> male2 = KeyValueProposition('Sex', Constraint.equals('male'))
    >>> female = KeyValueProposition('Sex', Constraint.equals('female'))
    >>> infant = KeyValueProposition('Age', Constraint.less_equals(4))
    >>> male == male2, male == infant
    (True, False)
    >>> male <= female, male >= female, infant <= female
    (False, True, True)
    """

    def __init__(self, key, constraint):
        self.key = key
        self.constraint = constraint
        self.repr = format(constraint, key)

    def __call__(self, row):
        return self.constraint(row[self.key])

    def __repr__(self):
        return self.repr

    def __eq__(self, other):
        return str(self) == str(other)

    def __le__(self, other):
        return str(self) <= str(other)


class TabulatedProposition:

    def __init__(self, table, col_idx):
        self.table = table
        self.col_idx = col_idx
        self.repr = 'c'+str(col_idx)

    def __call__(self, row_idx):
        return self.table[row_idx][self.col_idx]

    def __repr__(self):
        return self.repr


class Conjunction:
    """
    Conjunctive aggregation of propositions.

    For example:

    >>> old = KeyValueProposition('age', Constraint.greater_equals(60))
    >>> male = KeyValueProposition('sex', Constraint.equals('male'))
    >>> high_risk = Conjunction([male, old])
    >>> stephanie = {'age': 30, 'sex': 'female'}
    >>> erika = {'age': 72, 'sex': 'female'}
    >>> ron = {'age': 67, 'sex': 'male'}
    >>> high_risk(stephanie), high_risk(erika), high_risk(ron)
    (False, False, True)

    Elements can be accessed via index and are sorted lexicographically.
    >>> high_risk
    age>=60 & sex==male
    >>> high_risk[0]
    age>=60
    >>> len(high_risk)
    2

    >>> high_risk2 = Conjunction([old, male])
    >>> high_risk == high_risk2
    True

    >>> titanic = pd.read_csv("../datasets/titanic/train.csv")
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    >>> male = KeyValueProposition('Sex', Constraint.equals('male'))
    >>> third_class = KeyValueProposition('Pclass', Constraint.greater_equals(3))
    >>> conj = Conjunction([male, third_class])
    >>> titanic.loc[conj]
         Survived  Pclass   Sex   Age  SibSp  Parch     Fare Embarked
    0           0       3  male  22.0      1      0   7.2500        S
    4           0       3  male  35.0      0      0   8.0500        S
    5           0       3  male   NaN      0      0   8.4583        Q
    7           0       3  male   2.0      3      1  21.0750        S
    12          0       3  male  20.0      0      0   8.0500        S
    ..        ...     ...   ...   ...    ...    ...      ...      ...
    877         0       3  male  19.0      0      0   7.8958        S
    878         0       3  male   NaN      0      0   7.8958        S
    881         0       3  male  33.0      0      0   7.8958        S
    884         0       3  male  25.0      0      0   7.0500        S
    890         0       3  male  32.0      0      0   7.7500        Q
    <BLANKLINE>
    [347 rows x 8 columns]
    """

    def __init__(self, props):
        self.props = sorted(props, key=str)
        self.repr = str.join(" & ", map(str, self.props)) if props else 'True'

    def __call__(self, x):
        # TODO: check performance of the logical_and.reduce implementation (with list materialization)
        if not self.props:
            return ones(len(x), dtype='bool')  # TODO: check if this is correct handling for scalar x
        return logical_and.reduce([p(x) for p in self.props])
        # res = ones(len(x), dtype='bool')
        # for p in self.props:
        #     res &= p(x)
        # return res
        #return all(map(lambda p: p(x), self.props))

    def __repr__(self):
        return self.repr

    def __getitem__(self, item):
        return self.props[item]

    def __len__(self):
        return len(self.props)

    def __eq__(self, other):
        """
        Checks equality of conjunctions based on string representation.
        """
        return str(self) == str(other)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
