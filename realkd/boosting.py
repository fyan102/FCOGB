from math import sqrt, inf

import numpy as np
import scipy
import scipy.optimize
from numpy import zeros_like, array, argsort, cumsum, eye
from numpy.linalg import inv
from pandas import qcut
from realkd.logic import Conjunction
from sklearn.base import BaseEstimator
from datetime import datetime
from realkd.rules import AdditiveRuleEnsemble, loss_function, Rule, SquaredLoss
from realkd.search import Context, GoldenRatioSearch


class ObjectFunction:
    def __init__(self, data, target, predictions, loss, reg, rules=None):
        self.loss = loss_function(loss)
        self.reg = reg
        predictions = zeros_like(
            target) if predictions is None else predictions
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
        raise NotImplementedError()

    def bound(self, ext):
        raise NotImplementedError()

    def search(self, method='greedy', verbose=False, **search_params):
        from realkd.search import search_methods
        ctx = Context.from_df(self.data, **search_params)
        if verbose >= 2:
            print(
                f'Created search context with {len(ctx.attributes)} attributes')
        return search_methods[method](ctx, self, self.bound, verbose=verbose, **search_params).run()


class GradientBoostingObjectiveMWG(ObjectFunction):
    def __init__(self, data, target, predictions=None, loss=SquaredLoss, reg=1.0, rules=None, **kwargs):
        super().__init__(data, target, predictions, loss, reg, rules)
        predictions = zeros_like(
            target) if predictions is None else predictions
        g = array(self.loss.g(target, predictions))
        h = np.ones_like(g)
        r = g
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
        return abs(g_q.sum())

    def bound(self, ext):
        m = len(ext)
        if m == 0:
            return -inf
        g_q = self.g[ext]
        num_pre = abs(cumsum(g_q))
        num_suf = abs(cumsum(g_q[::-1]))
        neg_bound = num_suf.max()
        pos_bound = num_pre.max()
        return max(neg_bound, pos_bound)

    def opt_weight(self, q):
        ext = self.data.loc[q].index
        g_q = self.g[ext]
        h_q = self.h[ext]
        return -g_q.sum() / (self.reg + h_q.sum())


class OrthogonalBoostingObjectiveOld(ObjectFunction):
    def __init__(self, data, target, predictions=None, loss=SquaredLoss, reg=1.0, rules=None, **kwargs):
        super().__init__(data, target, predictions, loss, reg, rules)
        self.rules = [] if rules is None else rules
        self.loss = loss_function(loss)
        self.reg = reg
        predictions = zeros_like(
            target) if predictions is None else predictions
        g = array(self.loss.g(target, predictions))

        r = g
        order = argsort(r)[::-1]
        self.g = g[order]
        self.h = np.ones_like(g)
        self.data = data.iloc[order].reset_index(drop=True)
        self.target = target.iloc[order].reset_index(drop=True)
        self.n = len(target)
        if len(rules) != 0:
            self.q_mat = np.column_stack(
                [rules[i].q(self.data) + np.zeros(len(self.data)) for i in range(len(rules))])

    def __call__(self, ext):
        if len(ext) == 0:
            return -inf
        if len(self.rules) == 0:
            g_q = self.g[ext]
            h_q = self.h[ext]
            return abs(g_q.sum()) / (self.reg + np.sqrt(h_q.sum()))
        q = zeros_like(self.g)
        q[ext] = 1
        q_mat = self.q_mat
        proj_mat = np.eye(self.n) - q_mat.dot(inv(q_mat.T.dot(q_mat))).dot(q_mat.T)
        gg = proj_mat @ self.g
        q_orthogonal = proj_mat @ q
        length = norm(q_orthogonal)
        obj = abs(self.g.dot(q_orthogonal / length)) if length > 1e-6 else 0
        return obj

    def bound(self, ext):
        """
        Temporately use this bounding function, need to change to a better one
        :param ext: the extent of a query
        :return: currently it is the maximum cumulate sum of the gradient, need to be
                 changed
        """
        m = len(ext)
        if m == 0:
            return -inf
        g_q = self.g[ext]
        h_q = self.h[ext]
        num_pre = np.abs(cumsum(g_q))
        num_suf = np.abs(cumsum(g_q[::-1]))
        den_pre = np.sqrt(cumsum(h_q)) + self.reg
        den_suf = np.sqrt(cumsum(h_q[::-1])) + self.reg
        neg_bound = (num_suf / den_suf).max()
        pos_bound = (num_pre / den_pre).max()
        return max(neg_bound, pos_bound)


class OrthogonalBoostingObjective1(ObjectFunction):
    def __init__(self, data, target, predictions=None, loss=SquaredLoss, reg=1.0, rules=None, **kwargs):
        super().__init__(data, target, predictions, loss, reg, rules)
        self.rules = [] if rules is None else rules
        self.loss = loss_function(loss)
        self.reg = reg
        predictions = zeros_like(
            target) if predictions is None else predictions
        g = array(self.loss.g(target, predictions))
        h = np.ones_like(g)
        self.n = len(target)
        r = g / h
        order = argsort(r)[::-1]
        self.g = g[order]
        self.h = h[order]
        self.data = data.iloc[order].reset_index(drop=True)
        self.target = target.iloc[order].reset_index(drop=True)
        if len(rules) != 0:
            q_mat = np.column_stack(
                [rules[i].q(self.data) + np.zeros(len(self.data)) for i in range(len(rules))])
            self.proj_mat = eye(self.n) - q_mat @ inv(q_mat.T @ q_mat) @ q_mat.T
        else:
            self.proj_mat = eye(self.n)
        self.g_orth = self.proj_mat @ self.g

    def __call__(self, ext):
        if len(ext) == 0:
            return -inf
        if len(self.rules) == 0:
            g_q = self.g[ext]
            h_q = self.h[ext]
            return abs(g_q.sum()) / sqrt(h_q.sum())
        # get the q vector
        q = zeros_like(self.g)
        q[ext] = 1
        # project q onto orthogonal space
        q_orthogonal = self.proj_mat @ q
        # g_orth = self.proj_mat @ self.g
        length = norm(q_orthogonal)

        if length > 1e-6:
            obj = abs(self.g_orth @ q_orthogonal) / length
        else:
            obj = 0
        return obj

    def bound(self, ext):
        """
        Temporately use this bounding function, need to change to a better one
        :param ext: the extent of a query
        :return: currently it is the maximum cumulate sum of the gradient, need to be
                 changed
        """
        m = len(ext)
        if m == 0:
            return -inf
        q = zeros_like(self.g)
        q[ext] = 1

        g_q = self.g[ext]
        h_q = self.h[ext]
        num_pre = abs(cumsum(g_q))
        num_suf = abs(cumsum(g_q[::-1]))
        q_orthogonal = self.proj_mat @ q
        length = norm(q_orthogonal)
        if length <= 1e-6:
            return 0
        neg_bound = (num_suf / length).max()
        pos_bound = (num_pre / length).max()
        return max(neg_bound, pos_bound)


class OrthogonalBoostingObjective(ObjectFunction):
    def __init__(self, data, target, predictions=None, loss=SquaredLoss, reg=1.0, rules=None, **kwargs):
        super().__init__(data, target, predictions, loss, reg, rules)
        self.rules = [] if rules is None else rules
        self.loss = loss_function(loss)
        self.reg = reg
        predictions = zeros_like(
            target) if predictions is None else predictions
        g = array(self.loss.g(target, predictions))
        self.n = len(target)
        r = g
        order = argsort(r)[::-1]
        self.g = g[order]
        self.data = data.iloc[order].reset_index(drop=True)
        self.target = target.iloc[order].reset_index(drop=True)
        if len(rules) != 0:
            orth_basis = kwargs['orth_basis']
            self.orth_basis = orth_basis[order]
        else:
            self.orth_basis = np.zeros(self.n)

    def __call__(self, ext):
        if len(ext) == 0:
            return -inf
        g_q = self.g[ext]
        if len(self.rules) == 0:
            h_q = self.h[ext]
            return abs(g_q.sum()) / sqrt(h_q.sum())
        length = self.fast_orth_norm(ext)
        if length > 1e-4:
            obj = abs(g_q.sum()) / length
        else:
            obj = 0
        return obj

    def bound(self, ext):
        m = len(ext)
        if m == 0:
            return -inf
        q = zeros_like(self.g)
        q[ext] = 1
        g_q = self.g[ext]
        num_pre = abs(cumsum(g_q))
        num_suf = abs(cumsum(g_q[::-1]))
        if len(self.rules) != 0:
            qs = self.fast_para_norms_prefix(ext)
            qs_neg = self.fast_para_norms_prefix(ext[::-1])
        else:
            qs = np.sqrt(np.arange(1, m + 1))
            qs_neg = qs
        den_pre = qs
        den_suf = qs_neg
        neg_bound = np.divide(num_suf, den_suf, out=np.zeros_like(num_suf), where=den_suf != 0).max()
        pos_bound = np.divide(num_pre, den_pre, out=np.zeros_like(num_suf), where=den_pre != 0).max()
        return max(neg_bound, pos_bound)

    def fast_orth_norm(self, ext):
        deltas = self.orth_basis[ext]
        length = len(ext)
        okqi = np.sum(deltas, axis=0)
        q_para_norms = (okqi ** 2).sum()
        q_orth_norms_sq = np.abs(length - q_para_norms)
        return np.sqrt(q_orth_norms_sq)

    def fast_para_norms_prefix(self, ext):
        deltas = self.orth_basis[ext]
        length = len(ext)
        okqi = np.cumsum(deltas, axis=0)
        q_para_norms = (okqi ** 2).sum(axis=1)
        q_orth_norms_sq = np.abs(np.arange(1, length + 1) - q_para_norms)
        q_orth_norms = np.sqrt(q_orth_norms_sq)
        return q_orth_norms

    def bound2(self, ext):
        """
        Temporately use this bounding function, need to change to a better one
        :param ext: the extent of a query
        :return: currently it is the maximum cumulate sum of the gradient, need to be
                 changed
        """
        m = len(ext)
        if m == 0:
            return -inf
        g_q = self.g[ext]
        num_pre = abs(cumsum(g_q))
        num_suf = abs(cumsum(g_q[::-1]))
        length = self.fast_orth_norm(ext)
        if length <= 1e-6:
            return 0
        neg_bound = (num_suf / length).max()
        pos_bound = (num_pre / length).max()
        return max(neg_bound, pos_bound)


class OrthogonalBoostingObjectiveSlow(ObjectFunction):
    def __init__(self, data, target, predictions=None, loss=SquaredLoss, reg=1.0, rules=None, **kwargs):
        super().__init__(data, target, predictions, loss, reg, rules)
        self.rules = [] if rules is None else rules
        self.loss = loss_function(loss)
        self.reg = reg
        predictions = zeros_like(
            target) if predictions is None else predictions
        g = array(self.loss.g(target, predictions))
        self.n = len(target)
        if len(rules) != 0:
            orth_basis = kwargs['orth_basis']
            proj_mat = eye(self.n) - orth_basis @ orth_basis.T
        else:
            proj_mat = eye(self.n)
        # g_orth = np.linalg.norm(proj_mat @ np.diag(g), axis=1)
        # lengths = np.linalg.norm(proj_mat, axis=1)
        r = g  # / lengths
        order = argsort(r)[::-1]
        self.g = g[order]
        self.data = data.iloc[order].reset_index(drop=True)
        self.target = target.iloc[order].reset_index(drop=True)
        if len(rules) != 0:
            orth_basis = kwargs['orth_basis']
            self.orth_basis = orth_basis[order]
            self.proj_mat = eye(self.n) - self.orth_basis @ self.orth_basis.T
        else:
            self.proj_mat = eye(self.n)
            self.orth_basis = np.zeros(self.n)

    def __call__(self, ext):
        if len(ext) == 0:
            return -inf
        g_q = self.g[ext]
        if len(self.rules) == 0:
            h_q = self.h[ext]
            return abs(g_q.sum()) / sqrt(h_q.sum())
        # get the q vector
        q = zeros_like(self.g)
        q[ext] = 1
        # # project q onto orthogonal space
        q_orthogonal = self.proj_mat @ q
        # g_orth = self.proj_mat @ self.g
        length = norm(q_orthogonal)
        if length > 1e-4:
            obj = abs(g_q.sum()) / length
        else:
            obj = 0
        return obj

    def bound(self, ext):
        m = len(ext)
        if m == 0:
            return -inf
        q = zeros_like(self.g)
        q[ext] = 1
        g_q = self.g[ext]
        num_pre = abs(cumsum(g_q))
        num_suf = abs(cumsum(g_q[::-1]))
        qq = np.cumsum(np.eye(self.n)[ext, :], axis=0).T
        qq_orth = self.proj_mat @ qq
        qq_orth_neg = self.proj_mat @ qq[::-1]
        qs = np.sqrt((qq_orth ** 2).sum(axis=0))
        qs_neg = np.sqrt((qq_orth_neg ** 2).sum(axis=0))
        den_pre = qs
        den_suf = qs_neg
        neg_bound = np.divide(num_suf, den_suf, out=np.zeros_like(num_suf), where=den_suf != 0).max()
        pos_bound = np.divide(num_pre, den_pre, out=np.zeros_like(num_suf), where=den_pre != 0).max()
        return max(neg_bound, pos_bound)


class OrthogonalBoostingObjectiveXGB(ObjectFunction):
    def __init__(self, data, target, predictions=None, loss=SquaredLoss, reg=1.0, rules=None, **kwargs):
        super().__init__(data, target, predictions, loss, reg, rules)
        self.rules = [] if rules is None else rules
        self.loss = loss_function(loss)
        self.reg = reg
        predictions = zeros_like(
            target) if predictions is None else predictions
        g = array(self.loss.g(target, predictions))
        h = array(self.loss.h(target, predictions))
        self.n = len(target)
        r = g / h
        order = argsort(r)[::-1]
        self.g = g[order]
        self.h = h[order]
        self.data = data.iloc[order].reset_index(drop=True)
        self.target = target.iloc[order].reset_index(drop=True)
        if len(rules) != 0:
            q_mat = np.column_stack(
                [rules[i].q(self.data) + np.zeros(len(self.data)) for i in range(len(rules))])
            self.proj_mat = eye(self.n) - q_mat @ inv(q_mat.T @ q_mat) @ q_mat.T

    def __call__(self, ext):
        if len(ext) == 0:
            return -inf
        if len(self.rules) == 0:
            g_q = self.g[ext]
            h_q = self.h[ext]
            return (g_q.sum() ** 2) / (h_q.sum() + self.reg)
        # get the q vector
        q = zeros_like(self.g)
        q[ext] = 1
        # project q onto orthogonal space
        q_orthogonal = self.proj_mat @ q
        length = norm(q_orthogonal)

        if length > 1e-6:
            if abs(self.h @ q_orthogonal + self.reg) > 1e-6:
                obj = (self.g @ q_orthogonal) ** 2 / (self.h @ q_orthogonal + self.reg)
            else:
                obj = (self.g @ q_orthogonal) ** 2
        else:
            obj = 0
        return obj

    def bound(self, ext):
        """
        Temporately use this bounding function, need to change to a better one
        :param ext: the extent of a query
        :return: currently it is the maximum cumulate sum of the gradient, need to be
                 changed
        """
        m = len(ext)
        if m == 0:
            return -inf
        q = zeros_like(self.g)
        q[ext] = 1
        g_q = self.g[ext]
        h_q = self.h[ext]
        num_pre = (cumsum(g_q)) ** 2 * 2
        num_suf = (cumsum(g_q)) ** 2 * 2
        den_pre = (cumsum(h_q) + self.reg)  # + self.reg
        den_suf = (cumsum(h_q[::-1]) + self.reg)  # + self.reg
        neg_bound = (num_suf / den_suf).max()
        pos_bound = (num_pre / den_pre).max()
        return max(neg_bound, pos_bound)


class ShorterGradientBoostingObjective:
    def __init__(self, data, target, predictions=None, loss=SquaredLoss, reg=1.0, rules=None, **kwargs):
        self.loss = loss_function(loss)
        self.reg = reg
        predictions = zeros_like(
            target) if predictions is None else predictions
        g = array(self.loss.g(target, predictions))
        h = array(self.loss.h(target, predictions))
        r = g / h
        order = argsort(r)[::-1]
        self.g = g[order]
        self.h = h[order]
        self.data = data.iloc[order].reset_index(drop=True)
        self.target = target.iloc[order].reset_index(drop=True)
        self.n = len(target)

    def __call__(self, ext, num_props):
        alpha = num_props + 1
        if len(ext) == 0:
            return -inf
        g_q = self.g[ext]
        h_q = self.h[ext]
        return g_q.sum() ** 2 / (2 * self.n * (self.reg * alpha + h_q.sum()))

    def bound(self, ext, num_props):
        alpha = num_props + 1
        m = len(ext)
        if m == 0:
            return -inf
        g_q = self.g[ext]
        h_q = self.h[ext]
        num_pre = cumsum(g_q) ** 2
        num_suf = cumsum(g_q[::-1]) ** 2
        den_pre = cumsum(h_q) + self.reg * alpha
        den_suf = cumsum(h_q[::-1]) + self.reg * alpha
        neg_bound = (num_suf / den_suf).max() / (2 * self.n)
        pos_bound = (num_pre / den_pre).max() / (2 * self.n)
        return max(neg_bound, pos_bound)

    def opt_weight(self, q):
        ext = self.data.loc[q].index
        alpha = len(q) + 1
        g_q = self.g[ext]
        h_q = self.h[ext]
        return -g_q.sum() / (self.reg * alpha + h_q.sum())

    def search(self, method='greedy', verbose=False, **search_params):
        from realkd.search import search_methods
        ctx = Context.from_df(self.data, **search_params)
        if verbose >= 2:
            print(
                f'Created search context with {len(ctx.attributes)} attributes')
        return search_methods[method](ctx, self, self.bound, verbose=verbose, **search_params).run()


class GradientBoostingObjectiveGPE(ObjectFunction):
    def __init__(self, data, target, predictions=None, loss=SquaredLoss, reg=1.0, rules=None, **kwargs):
        super().__init__(data, target, predictions, loss, reg, rules)
        self.loss = loss_function(loss)
        self.reg = reg
        predictions = zeros_like(
            target) if predictions is None else predictions
        g = array(self.loss.g(target, predictions))
        h = np.ones_like(g)
        r = g
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
        return abs(g_q.sum()) / (np.sqrt(h_q.sum())) / (2 * self.n)

    def bound(self, ext):
        m = len(ext)
        if m == 0:
            return -inf
        g_q = self.g[ext]
        h_q = self.h[ext]
        num_pre = np.abs(cumsum(g_q))
        num_suf = np.abs(cumsum(g_q[::-1]))
        den_pre = np.sqrt(cumsum(h_q))  # + self.reg
        den_suf = np.sqrt(cumsum(h_q[::-1]))  # + self.reg
        neg_bound = (num_suf / den_suf).max() / (2 * self.n)
        pos_bound = (num_pre / den_pre).max() / (2 * self.n)
        return max(neg_bound, pos_bound)

    def opt_weight(self, q):
        ext = self.data.loc[q].index
        g_q = self.g[ext]
        h_q = self.h[ext]
        return -g_q.sum() / (self.reg + h_q.sum())


def norm(x):
    """
    Calculate the L-2 norm of a vector

    :param x: the vector whose L-2 norm is to be calculated
    :return: the L-2 norm of the vector
    """
    return (x * x).sum() ** 0.5


class WeightUpdateMethod:
    def __init__(self, loss, reg=1.0):
        self.loss = loss
        self.reg = reg

    def calc_weight(self, data, target, rules):
        raise NotImplementedError()


class FullyCorrective(WeightUpdateMethod):
    def __init__(self, loss='squared', reg=1.0, solver='Newton-CG'):
        super().__init__(loss, reg)
        self.solver = solver

    @staticmethod
    def get_risk(loss, y, q_mat, reg):
        def sum_loss(weights):
            return sum(loss(y, q_mat.dot(weights))) + reg * sum(weights * weights) / 2

        return sum_loss

    @staticmethod
    def get_gradient(g, y, q_mat, reg):
        def gradient(weights):
            grad_vec = g(y, q_mat.dot(weights))
            return q_mat.T.dot(grad_vec) + reg * weights

        return gradient

    @staticmethod
    def get_hessian(h, y, q_mat, reg):
        def hessian(weights):
            h_vec = h(y, q_mat.dot(weights))
            return q_mat.T.dot(np.diag(h_vec)).dot(q_mat) + np.diag([reg] * len(weights))

        return hessian

    def calc_weight(self, data, target, rules):
        g = loss_function(self.loss).g
        h = loss_function(self.loss).h
        loss = loss_function(self.loss)
        y = np.array(target)
        q_mat = np.column_stack(
            [rules[i].q(data) + np.zeros(len(data)) for i in range(len(rules))])
        sum_loss = self.get_risk(loss, y, q_mat, self.reg)
        gradient = self.get_gradient(g, y, q_mat, self.reg)
        hessian = self.get_hessian(h, y, q_mat, self.reg)

        if self.solver == 'GD':  # Gradient descent
            w = np.array([r.y for r in rules])
            old_w = np.ones_like(w) * (1.0 if len(w) - sum(w) > 1e-5 else 2.0)
            i = 0
            while norm(old_w - w) > 1e-3 and i < 100:
                old_w = np.array(w)
                if norm(gradient(w)) == 0:
                    break
                p = -gradient(w) / norm(gradient(w))
                w += GoldenRatioSearch(sum_loss, old_w, p, gradient).run() * p
                i += 1
        elif self.solver == 'Line':
            w = np.array([r.y for r in rules])
            if norm(gradient(w)) != 0:
                p = -gradient(w) / norm(gradient(w))
                distance = GoldenRatioSearch(sum_loss, w, p, gradient).run()
                w += distance * p
        else:
            w = np.array([r.y for r in rules])
            w = scipy.optimize.minimize(sum_loss, w, method=self.solver, jac=gradient, hess=hessian,
                                        options={'disp': False}).x
        return w


class LineSearch(WeightUpdateMethod):
    def __init__(self, loss='squared', reg=1.0):
        super().__init__(loss, reg)

    @staticmethod
    def get_risk(loss, y, q_mat, weights: np.array, reg):
        def sum_loss(weight):
            all_weights = np.append(weights, weight)
            return sum(loss(y, q_mat.dot(all_weights))) + reg * sum(all_weights * all_weights) / 2

        return sum_loss

    @staticmethod
    def get_gradient(g, y, q_mat, weights: np.array, reg):
        def gradient(weight):
            all_weights = np.append(weights, weight)
            grad_vec = g(y, q_mat.dot(all_weights))
            return np.array([(q_mat.T.dot(grad_vec) + reg * all_weights)[-1]])

        return gradient

    def calc_weight(self, data, target, rules):
        w = np.array([rules[-1].y])
        all_weights = np.array([rule.y for rule in rules][:-1])
        loss = loss_function(self.loss)
        g = loss_function(self.loss).g
        y = np.array(target)
        q_mat = np.column_stack(
            [rules[i].q(data) + np.zeros(len(data)) for i in range(len(rules))])
        sum_loss = self.get_risk(loss, y, q_mat, all_weights, self.reg)
        gradient = self.get_gradient(g, y, q_mat, all_weights, self.reg)
        if norm(gradient(w)) != 0:
            p = -gradient(w) / norm(gradient(w))
            distance = GoldenRatioSearch(sum_loss, w, p, gradient).run()
            w += distance * p
        all_weights = np.append(all_weights, w)
        return all_weights


class KeepWeight(WeightUpdateMethod):
    def __init__(self, loss='squared', reg=1.0):
        super().__init__(loss, reg)

    def calc_weight(self, data, target, rules):
        all_weights = np.array([rule.y for rule in rules])
        return all_weights
def orthonormalization(Q):
    n, k = Q.shape
    O = np.zeros(shape=(n, k))
    q = Q[:, 0]
    O[:, 0] =  q / norm(q)

    for i in range(1, k):
        O_i = O[:, :i]
        q = Q[:, i]
        q_orth = q - O_i.dot(O_i.T.dot(q))
        O[:, i] =  q_orth / norm(q_orth)
    return O

class GeneralRuleBoostingEstimator(BaseEstimator):
    def __init__(self, num_rules, objective_function, weight_update_method, loss='squared', reg=1.0,
                 search='exhaustive', max_col_attr=10,
                 search_params=None, verbose=False):
        if search_params is None:
            search_params = {'order': 'bestboundfirst', 'apx': 1.0, 'max_depth': None, 'discretization': qcut,
                             'max_col_attr': max_col_attr}
        self.num_rules = num_rules
        self.objective = objective_function
        self.objective_function = objective_function
        self.max_col_attr = max_col_attr
        self.weight_update_method = weight_update_method
        self.loss = loss_function(loss)
        self.reg = reg
        self.weight_update_method.loss = loss
        self.weight_update_method.reg = reg
        self.verbose = verbose
        self.search = search
        self.rules_ = AdditiveRuleEnsemble([])
        self.search_params = search_params
        self.history = []
        self.time = []

    def set_reg(self, reg):
        self.reg = reg
        self.objective.reg = reg
        self.weight_update_method.reg = reg

    def fit(self, data, target, has_origin_rules=False, verbose=False):
        self.history = []
        self.time = []
        if not has_origin_rules:
            self.rules_.members = []
            orth_basis = np.array([])
        else:
            q_mat = np.column_stack(
                [self.rules_[i].q(data) + np.zeros(len(data)) for i in range(len(self.rules_))])
            orth_basis = orthonormalization(q_mat)

        while len(self.rules_) < self.num_rules:
            start_time = datetime.now()
            # Search for a rule
            scores = self.rules_(data)
            obj = self.objective(data, target, predictions=scores,
                                 loss=self.loss, reg=self.reg, rules=self.rules_, orth_basis=orth_basis)
            q = obj.search(method=self.search, verbose=verbose,
                           **self.search_params)
            if hasattr(self.objective, 'opt_weight') and callable(getattr(self.objective, 'opt_weight')):
                y = obj.opt_weight(q)
            else:
                y = 1.0
            q_vec = q(data)

            if len(orth_basis) == 0:
                basis = q_vec / norm(q_vec)
                orth_basis = np.array([basis]).T
            else:
                basis = q_vec - orth_basis.dot(orth_basis.T.dot(q_vec))
                basis = basis / norm(basis)
                orth_basis = np.hstack((orth_basis, np.array([basis]).T))
            rule = Rule(q, y)
            if self.verbose:
                print(rule)
            self.rules_.append(rule)
            # Calculate weights
            weights = self.weight_update_method.calc_weight(
                data, target, self.rules_)
            for i in range(len(self.rules_)):
                self.rules_[i].y = weights[i]
            self.history.append(AdditiveRuleEnsemble(
                [Rule(q=rule.q, y=rule.y) for rule in self.rules_.members]))
            end_time = datetime.now()
            self.time.append(str(end_time - start_time))
        return self

    def predict(self, data):
        loss = loss_function(self.loss)
        return loss.preidictions(self.rules_(data))

    def decision_function(self, data):
        return self.rules_(data)
