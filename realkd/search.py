"""
Methods for searching for conjunctions in a binary (formal) search context.
"""
import inspect

import pandas as pd
import sortednp as snp
import doctest

from collections import defaultdict, deque
from sortedcontainers import SortedSet
from math import inf, sqrt
from heapq import heappop, heappush
from numpy import array
from bitarray import bitarray
from bitarray.util import subset

from realkd.logic import Conjunction, Constraint, KeyValueProposition, TabulatedProposition


class Node:
    """
    Represents a potential node (and incoming edge) for searches in the concept graph
    with edges representing the direct prefix-preserving successor relation (dpps).
    """

    def __init__(self, gen, clo, ext, bit_ext, idx, crit_idx, val, bnd):
        self.generator = gen
        self.closure = clo
        self.extension = ext
        self.bit_extension = bit_ext
        self.gen_index = idx
        self.crit_idx = crit_idx
        self.val = val
        self.val_bound = bnd
        self.valid = self.crit_idx > self.gen_index

    def __repr__(self):
        return f'N({list(self.generator)}, {array([i for i in range(len(self.closure)) if self.closure[i]])}, {self.val:.5g}, {self.val_bound:.5g}, {self.extension})'

    def value(self):
        return self.val

    def __le__(self, other):
        return self.val <= other.val

    def __eq__(self, other):
        return self.val == other.val

    def __ge__(self, other):
        return self.val >= other.val

    def __lt__(self, other):
        return self.val < other.val

    def __gt__(self, other):
        return self.val > other.val


class BreadthFirstBoundary:

    def __init__(self):
        self.deq = deque()

    def __bool__(self):
        return bool(self.deq)

    def __len__(self):
        return len(self.deq)

    def push(self, augmented_node):
        self.deq.append(augmented_node)

    def pop(self):
        return self.deq.popleft()


class DepthFirstBoundary:

    def __init__(self):
        self.stack = []

    def __bool__(self):
        return bool(self.stack)

    def __len__(self):
        return len(self.stack)

    def push(self, augmented_node):
        self.stack.append(augmented_node)

    def pop(self):
        return self.stack.pop()


class BestBoundFirstBoundary:

    def __init__(self):
        self.heap = []

    def __bool__(self):
        return bool(self.heap)

    def __len__(self):
        return len(self.heap)

    def push(self, augmented_node):
        _, node = augmented_node
        heappush(self.heap, (-node.val_bound, -node.val, augmented_node))

    def pop(self):
        _, _, augmented_node = heappop(self.heap)
        return augmented_node


class BestValueFirstBoundary:

    def __init__(self):
        self.heap = []

    def __bool__(self):
        return bool(self.heap)

    def __len__(self):
        return len(self.heap)

    def push(self, augmented_node):
        _, node = augmented_node
        heappush(self.heap, (-node.val, -node.val_bound, augmented_node))

    def pop(self):
        _, _, augmented_node = heappop(self.heap)
        return augmented_node


class Context:
    """
    Formal context, i.e., a binary relation between a set of objects and a set of attributes,
    i.e., Boolean functions that are defined on the objects.

    A formal context provides a search context (search space) over conjunctions that can be
    formed from the individual attributes.
    """

    @staticmethod
    def from_tab(table, sort_attributes=False):
        """
        Converts an input table where each row represents an object into
        a formal context (which uses column-based representation).

        Uses Boolean interpretation of table values to determine attribute
        presence for an object.

        For instance:

        >>> table = [[0, 1, 0, 1],
        ...          [1, 1, 1, 0],
        ...          [1, 0, 1, 0],
        ...          [0, 1, 0, 1]]
        >>> ctx = Context.from_tab(table)
        >>> list(ctx.extension([0, 2]))
        [1, 2]

        :param table: table that explicitly encodes object/attribute relation
        :return: context over table row indices as objects and one tabulated feature per column index
        """

        m = len(table)
        n = len(table[0])
        attributes = [TabulatedProposition(table, j) for j in range(n)]
        return Context(attributes, list(range(m)), sort_attributes)

    @staticmethod
    def from_df(df, without=None, max_col_attr=10, sort_attributes=True, discretization=pd.qcut, **kwargs):
        """
        Generates formal context from pandas dataframe by applying inter-ordinal scaling to numerical data columns
        and for object columns creating one attribute per value.

        For inter-ordinal scaling a maximum number of attributes per column can be specified. If required, threshold
        values are then selected by the provided discretization function (per default quantile-based).

        The restriction should also be implemented for object columns in the future (by merging small categories
        into disjunctive propositions).

        The generated attributes correspond to pandas-compatible query strings. For example:

        >>> titanic_df = pd.read_csv("../datasets/titanic/train.csv")
        >>> titanic_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
        >>> titanic_ctx = Context.from_df(titanic_df, max_col_attr=6, sort_attributes=False)
        >>> titanic_ctx.m
        891
        >>> titanic_ctx.attributes # doctest: +NORMALIZE_WHITESPACE
        [Survived<=0, Survived>=1, Pclass<=1, Pclass<=2, Pclass>=2, Pclass>=3, Sex==male, Sex==female, Age<=23.0,
        Age>=23.0, Age<=34.0, Age>=34.0, Age<=80.0, Age>=80.0, SibSp<=8.0, SibSp>=8.0, Parch<=6.0, Parch>=6.0,
        Fare<=8.6625, Fare>=8.6625, Fare<=26.0, Fare>=26.0, Fare<=512.3292, Fare>=512.3292, Embarked==S, Embarked==C,
        Embarked==Q, Embarked==nan]
        >>> titanic_ctx.n
        28
        >>> titanic_df.query('Survived>=1 & Pclass>=3 & Sex=="male" & Age>=34')
             Survived  Pclass   Sex   Age  SibSp  Parch   Fare Embarked
        338         1       3  male  45.0      0      0  8.050        S
        400         1       3  male  39.0      0      0  7.925        S
        414         1       3  male  44.0      0      0  7.925        S
        >>> titanic_ctx.extension([1, 5, 6, 11])
        array([338, 400, 414])

        >>> titanic_ctx = Context.from_df(titanic_df, max_col_attr=defaultdict(lambda: None, Age=6, Fare=6),
        ...                               sort_attributes=False)
        >>> titanic_ctx.attributes # doctest: +NORMALIZE_WHITESPACE
        [Survived<=0, Survived>=1, Pclass<=1, Pclass<=2, Pclass>=2, Pclass>=3, Sex==male, Sex==female, Age<=23.0,
        Age>=23.0, Age<=34.0, Age>=34.0, Age<=80.0, Age>=80.0, SibSp<=0, SibSp<=1, SibSp>=1, SibSp<=2, SibSp>=2,
        SibSp<=3, SibSp>=3, SibSp<=4, SibSp>=4, SibSp<=5, SibSp>=5, SibSp>=8, Parch<=0, Parch<=1, Parch>=1, Parch<=2,
        Parch>=2, Parch<=3, Parch>=3, Parch<=4, Parch>=4, Parch<=5, Parch>=5, Parch>=6, Fare<=8.6625, Fare>=8.6625,
        Fare<=26.0, Fare>=26.0, Fare<=512.3292, Fare>=512.3292, Embarked==S, Embarked==C, Embarked==Q, Embarked==nan]


        :param DataFrame df: pandas dataframe to be converted to formal context
        :param int max_col_attr: maximum number of attributes generated per column;
                             or None if an arbitrary number of attributes is permitted;
                             or dict (usually defaultdict) with keys being columns ids of df and values
                             being the maximum number of attributes for the corresponding column (again using
                             None if no bound for a specific column);
                             Note: use defaultdict(lambda: None) instead of defaultdict(None) to specify no maximum
                             per default
        :param callable discretization: the discretization function to be used when number of thresholds has to be reduced to
                               a specificed maximum (function has to have identical signature to pandas.qcut, which
                               is the default)
        :param Iterable[str] without: columns to ommit
        :return: :class:`Context` representing dataframe
        """

        without = without or []

        if not isinstance(max_col_attr, dict):
            const = max_col_attr
            max_col_attr = defaultdict(lambda: const)

        attributes = []
        for c in df:
            if c in without:
                continue
            if df[c].dtype.kind in 'uif':
                vals = df[c].unique()
                reduced = False
                max_cols = max_col_attr[str(c)]
                if max_cols and len(vals) * 2 > max_cols:
                    _, vals = discretization(
                        df[c], max_cols // 2, retbins=True, duplicates='drop')
                    vals = vals[1:]
                    reduced = True
                vals = sorted(vals)
                for i, v in enumerate(vals):
                    if reduced or i < len(vals) - 1:
                        attributes += [KeyValueProposition(
                            c, Constraint.less_equals(v))]
                    if reduced or i > 0:
                        attributes += [KeyValueProposition(
                            c, Constraint.greater_equals(v))]
            if df[c].dtype.kind in 'O':
                attributes += [KeyValueProposition(c, Constraint.equals(v))
                               for v in df[c].unique()]

        return Context(attributes, [df.iloc[i] for i in range(len(df.axes[0]))], sort_attributes)

    def __init__(self, attributes, objects, sort_attributes=True):
        self.attributes = attributes
        self.objects = objects
        self.n = len(attributes)
        self.m = len(objects)
        # for now we materialise the whole binary relation; in the future can be on demand
        # self.extents = [SortedSet([i for i in range(self.m) if attributes[j](objects[i])]) for j in range(self.n)]
        self.extents = [array([i for i in range(self.m) if attributes[j](
            objects[i])], dtype='int64') for j in range(self.n)]
        self.bit_extents = [bitarray([True if attributes[j](
            objects[i]) else False for i in range(self.m)]) for j in range(self.n)]

        # sort attribute in ascending order of extent size
        if sort_attributes:
            attribute_order = list(
                sorted(range(self.n), key=lambda i: len(self.extents[i])))
            self.attributes = [self.attributes[i] for i in attribute_order]
            self.extents = [self.extents[i] for i in attribute_order]
            self.bit_extents = [self.bit_extents[i] for i in attribute_order]

        # switches
        self.crit_propagation = True

        # stats
        self.popped = 0
        self.created = 0
        self.avg_created_length = 0
        self.rec_crit_hits = 0
        self.crit_hits = 0
        self.del_bnd_hits = 0
        self.clo_hits = 0
        self.non_lexmin_hits = 0
        self.bnd_post_children_hits = 0
        self.bnd_immediate_hits = 0

    def greedy_simplification(self, intent, extent):
        to_cover = SortedSet([i for i in range(self.m) if i not in extent])
        available = list(range(len(intent)))
        covering = [SortedSet(
            [i for i in range(self.m) if i not in self.extents[j]]) for j in intent]
        result = []
        while to_cover:
            j = max(available, key=lambda i: len(covering[i]))
            result += [intent[j]]
            available.remove(j)
            to_cover -= covering[j]
            for l in available:
                covering[l] -= covering[j]

        return result

    def extension(self, intent):
        """
        :param intent: attributes describing a set of objects
        :return: indices of objects that have all attributes in intent in common
        """
        if not intent:
            return array(range(len(self.objects)))

        result = self.extents[intent[0]]
        for i in range(1, len(intent)):
            result = snp.intersect(result, self.extents[intent[i]])

        return result

    def find_small_crit_index(self, gen_idx, bit_extension, part_closure):
        """
        >>> table = [[0, 1, 0, 1],
        ...          [1, 1, 1, 0],
        ...          [1, 0, 1, 0],
        ...          [0, 1, 0, 1]]
        >>> ctx = Context.from_tab(table)
        >>> root = Node([], bitarray('0000'), array([0,1,2,3]), bitarray('1111'), -1, -4, 1, inf)
        >>> ctx.find_small_crit_index(0, bitarray('0110'), bitarray('1000'))
        4
        >>> ctx.find_small_crit_index(1, bitarray('1101'), bitarray('0100'))
        4
        >>> ctx.find_small_crit_index(2, bitarray('0110'), bitarray('0010'))
        0
        >>> ctx.find_small_crit_index(3, bitarray('1001'), bitarray('0001'))
        1
        >>> ctx.find_small_crit_index(3, bitarray('1001'), bitarray('0101'))
        4
        """
        for j in range(0, gen_idx):
            # and len(extension) <= len(self.extents[j])
            if not part_closure[j] and subset(bit_extension, self.bit_extents[j]):
                return j
        return len(part_closure)

    def complete_closure(self, gen_idx, bit_extension, part_closure):
        """
        :param gen_idx:
        :param bit_extension:
        :param closure_prefix:
        :return:

        >>> table = [[0, 1, 0, 1],
        ...          [1, 1, 1, 0],
        ...          [1, 0, 1, 0],
        ...          [0, 1, 0, 1]]
        >>> ctx = Context.from_tab(table)
        >>> clo = bitarray('1000')
        >>> ctx.complete_closure(0, bitarray('0110'), clo)
        2
        >>> clo
        bitarray('1010')
        >>> clo = bitarray('1110')
        >>> ctx.complete_closure(1, bitarray('0100'), clo)
        4
        >>> clo
        bitarray('1110')
        """
        n = len(part_closure)
        crit_idx = n
        for j in range(gen_idx + 1, n):
            # TODO: for the moment put guard out because it seems faster, but this could change with
            #       numba and/or be different for different datasets
            # and len(extension) <= len(self.extents[j])
            if not part_closure[j] and subset(bit_extension, self.bit_extents[j]):
                crit_idx = min(crit_idx, j)
                part_closure[j] = True

        return crit_idx


class CoreQueryTreeSearch:
    r"""Searches the prefix-tree of core queries given a certain :class:`Context`.

    The idea of core queries is that there is only one core query per potential query extensions. Thus using them
    as solution candidates results in a strongly condensed search space compared to naively searching all conjunctions.

    Core queries have been originally introduced for closed itemset mining :cite:`uno2004efficient`.
    The following is a simple recursive definition that does not require the notion of closures:

    The trivial query :math:`q = \top` is a core query. Moreover, the tail augmentation :math:`qp_i` of a core query
    :math:`q` is also a core query if :math:`q \not\rightarrow p_i` and

    .. math::
        :nowrap:

        \begin{equation}
        \text{for all } j < i, \text{ if } q' \rightarrow p_j \text{ then } q \rightarrow p_j \enspace .
        \end{equation}

    """

    #: dictionary of available traversal orders for core query search
    traversal_orders = {
        'breadthfirst': BreadthFirstBoundary,
        'bestboundfirst': BestBoundFirstBoundary,
        'bestvaluefirst': BestValueFirstBoundary,
        'depthfirst': DepthFirstBoundary
    }

    def __init__(self, ctx, obj, bnd, order='bestboundfirst', apx=1.0, max_depth=10, verbose=False,
                 **kwargs):
        """

        :param Context ctx: the context defining the search space
        :param callable obj: objective function
        :param callable bnd: bounding function satisfying that ``bnd(q) >= max{obj(r) for r in successors(q)}``
        :param str order: traversal order (``'breadthfirst'``, ``'depthfirst'``, ``'bestboundfirst'``, or ``'bestvaluefirst'``; see :data:`traversal_orders`)
        :param float apx: approximation factor that determines guarantee of what fraction of search space will be traversed, i.e., all nodes q are visited with ``obj(q) >= apx * opt`` (default ``1.0``, i.e., optimum will be visited; smaller values means less traversal elements
        :param int max_depth: maximum depth of explored search nodes
        :param int verbose: level of verbosity

        """
        self.ctx = ctx
        self.f = obj
        self.g = bnd
        self.order = order
        self.apx = apx
        self.max_depth = max_depth
        self.verbose = verbose

        # switches
        self.crit_propagation = True

        # stats
        self.popped = 0
        self.created = 0
        self.avg_created_length = 0
        self.rec_crit_hits = 0
        self.crit_hits = 0
        self.del_bnd_hits = 0
        self.clo_hits = 0
        self.non_lexmin_hits = 0
        self.bnd_post_children_hits = 0
        self.bnd_immediate_hits = 0
        self.obj_para_num = len(inspect.getfullargspec(self.f).args)

    def traversal(self):
        """
        A first example with trivial objective and bounding function is as follows. In this example
        the optimal extension is the empty extension, which is generated via the
        the non-lexicographically smallest generator [0, 3].

        >>> table = [[0, 1, 0, 1],
        ...          [1, 1, 1, 0],
        ...          [1, 0, 1, 0],
        ...          [0, 1, 0, 1]]
        >>> ctx = Context.from_tab(table)
        >>> search = CoreQueryTreeSearch(ctx, lambda e: -len(e), lambda e: 0, order='breadthfirst')
        >>> for n in search.traversal():
        ...     print(n)
        N([], [], -4, inf, [0 1 2 3])
        N([0], [0 2], -2, 0, [1 2])
        N([1], [1], -3, 0, [0 1 3])
        N([2], [0 2], -2, 0, [1 2])
        N([3], [1 3], -2, 0, [0 3])
        N([0, 1], [0 1 2], -1, 0, [1])
        N([0, 3], [0 1 2 3], 0, 0, [])

        >>> search.verbose = True
        >>> search.run()
        <BLANKLINE>
        Found optimum after inspecting 7 nodes: [0, 3]
        Completing closure
        Greedy simplification: [0, 3]
        c0 & c3

        >>> CoreQueryTreeSearch(ctx, lambda e: 5-len(e), lambda e: 4+(len(e)>=2), apx=0.7).run()
        c0 & c1

        Let's use more realistic objective and bounding functions based on values associated with each
        object (row in the table).

        >>> values = [-1, 1, 1, -1]
        >>> f = lambda e: sum((values[i] for i in e))/4
        >>> g = lambda e: sum((values[i] for i in e if values[i]==1))/4
        >>> search = CoreQueryTreeSearch(ctx, f, g)
        >>> for n in search.traversal():
        ...     print(n)
        N([], [], 0, inf, [0 1 2 3])
        N([0], [0 2], 0.5, 0.5, [1 2])
        N([2], [0 2], 0.5, 0.5, [1 2])

        Finally, here is a complex example taken from the UdS seminar on subgroup discovery.

        >>> table = [[1, 1, 1, 1, 0],
        ...          [1, 1, 0, 0, 0],
        ...          [1, 0, 1, 0, 0],
        ...          [0, 1, 1, 1, 1],
        ...          [0, 0, 1, 1, 1],
        ...          [1, 1, 0, 0, 1]]
        >>> ctx = Context.from_tab(table)
        >>> labels = [1, 0, 1, 0, 0, 0]
        >>> from realkd.legacy import impact, cov_incr_mean_bound, impact_count_mean
        >>> f = impact(labels)
        >>> g = cov_incr_mean_bound(labels, impact_count_mean(labels))
        >>> search = CoreQueryTreeSearch(ctx, f, g)
        >>> for n in search.traversal():
        ...     print(n)
        N([], [], 0, inf, [0 1 2 3 4 5])
        N([0], [0], 0.11111, 0.22222, [0 1 2 5])
        N([1], [1], -0.055556, 0.11111, [0 1 3 5])
        N([2], [2], 0.11111, 0.22222, [0 2 3 4])
        N([3], [2 3], 0, 0.11111, [0 3 4])
        N([0, 2], [0 2], 0.22222, 0.22222, [0 2])
        >>> search.run()
        c0 & c2

        >>> labels = [1, 0, 0, 1, 1, 0]
        >>> f = impact(labels)
        >>> g = cov_incr_mean_bound(labels, impact_count_mean(labels))
        >>> search = CoreQueryTreeSearch(ctx, f, g)
        >>> for n in search.traversal():
        ...     print(n)
        N([], [], 0, inf, [0 1 2 3 4 5])
        N([0], [0], -0.16667, 0.083333, [0 1 2 5])
        N([1], [1], 0, 0.16667, [0 1 3 5])
        N([2], [2], 0.16667, 0.25, [0 2 3 4])
        N([3], [2 3], 0.25, 0.25, [0 3 4])
        """
        boundary = CoreQueryTreeSearch.traversal_orders[self.order]()
        full = self.ctx.extension([])
        full_bits = bitarray(len(full))
        full_bits.setall(1)
        if self.obj_para_num == 2:
            root = Node(SortedSet([]), bitarray((0 for _ in range(self.ctx.n))), full, full_bits, -1, self.ctx.n,
                        self.f(full), inf)
        else:
            root = Node(SortedSet([]), bitarray((0 for _ in range(self.ctx.n))), full, full_bits, -1, self.ctx.n,
                        self.f(full, 0), inf)
        opt = root
        yield root

        boundary.push(([(i, self.ctx.n, inf)
                        for i in range(self.ctx.n)], root))
        self.created += 1

        while boundary:
            ops, current = boundary.pop()

            self.popped += 1

            if self.verbose >= 2 and self.popped % 1000 == 0:
                print('*', end='', flush=True)
            if self.verbose >= 1 and self.popped % 10000 == 0:
                print(f' (lwr/upp/rat: {opt.val:.4f}/{current.val_bound:.4f}/{opt.val / current.val_bound:.4f},'
                      f' opt/avg depth: {len(opt.generator)}/{self.avg_created_length:.2f},'
                      f' bndry: {len(boundary)})', flush=True)

            children = []
            # for a in ops:
            for aug, crit, bnd in ops:
                if aug <= current.gen_index:  # need to also check == case it seems
                    continue
                if self.crit_propagation and crit < current.gen_index:
                    self.rec_crit_hits += 1
                    continue
                if bnd * self.apx <= opt.val:  # checking old bound against potentially updated opt value
                    self.del_bnd_hits += 1
                    continue
                if current.closure[aug]:
                    self.clo_hits += 1
                    continue

                extension = snp.intersect(
                    current.extension, self.ctx.extents[aug])
                generator = current.generator[:]
                generator.append(aug)
                if self.obj_para_num == 2:
                    val = self.f(extension)
                    bound = self.g(extension)
                else:
                    val = self.f(extension, len(generator))
                    bound = self.g(extension, len(generator))
                if bound * self.apx < opt.val and val <= opt.val:
                    self.bnd_immediate_hits += 1
                    continue

                self.created += 1
                self.avg_created_length = self.avg_created_length * ((self.created - 1) / self.created) + \
                                          len(generator) / self.created
                bit_extension = current.bit_extension & self.ctx.bit_extents[aug]
                closure = bitarray(current.closure)
                closure[aug] = True

                if self.crit_propagation and crit < aug and not current.closure[crit]:
                    # aug still needed for descendants but for current is guaranteed
                    # to lead to not lexmin child; hence can recycle current crit index
                    # (as upper bound to real crit index)
                    self.crit_hits += 1
                    crit_idx = crit
                else:
                    crit_idx = self.ctx.find_small_crit_index(
                        aug, bit_extension, closure)

                if crit_idx > aug:  # in this case crit_idx == n (sentinel)
                    crit_idx = self.ctx.complete_closure(
                        aug, bit_extension, closure)
                else:
                    closure[crit_idx] = True
                child = Node(generator, closure, extension,
                             bit_extension, aug, crit_idx, val, bound)
                opt = max(opt, child, key=Node.value)
                yield child

                # early termination if opt value approximately exceeds best active upper bound
                if opt.val >= self.apx * current.val_bound and self.order == 'bestboundfirst':
                    if self.verbose:
                        print(
                            f'best value {opt.val:.4f} {self.apx}-apx. exceeds best active bound {current.val_bound:.4f}')
                        print(f'terminating traversal')
                    return

                children += [child]

            augs = []
            for child in children:
                if child.val_bound * self.apx > opt.val:
                    augs.append(
                        (child.gen_index, child.crit_idx, child.val_bound))
                else:
                    self.bnd_post_children_hits += 1

            for child in children:
                if child.valid and (not self.max_depth or len(child.generator) < self.max_depth):
                    boundary.push((augs, child))
                else:
                    self.non_lexmin_hits += 1

    def print_stats(self):
        print()
        print('Pruning rule hits')
        print('-----------------')
        print('bound propagation   (sgl):', self.del_bnd_hits)
        print('crit propagation    (rec):', self.rec_crit_hits)
        print('crit propagation    (sgl):', self.crit_hits)
        print('crit violation      (rec):', self.non_lexmin_hits)
        print('equivalence         (rec):', self.clo_hits)
        print('bnd immediate       (rec):', self.bnd_immediate_hits)
        print('bnd post children   (rec):', self.bnd_post_children_hits)

    def run(self):
        """
        Runs the configured search.

        :return: :class:`~realkd.logic.Conjunction` that (approximately) maximizes objective

        """
        if self.verbose >= 2:
            print(
                f'Searching with apx factor {self.apx} and depth limit {self.max_depth} in order {self.order}')
        opt = None
        opt_value = -inf
        k = 0
        for node in self.traversal():
            k += 1
            if opt_value < node.val:
                opt = node
                opt_value = node.val
        if self.verbose:
            print('')
            print(f'Found optimum after inspecting {k} nodes: {opt.generator}')

        if self.verbose >= 3:
            self.print_stats()

        if opt and not opt.valid:
            if self.verbose:
                print('Completing closure')
            self.ctx.complete_closure(
                opt.gen_index, opt.bit_extension, opt.closure)
        min_generator = self.ctx.greedy_simplification([i for i in range(len(opt.closure)) if opt.closure[i]],
                                                       opt.extension)
        if self.verbose:
            print('Greedy simplification:', min_generator)
        return Conjunction(map(lambda i: self.ctx.attributes[i], min_generator))


class GreedySearch:
    """
    Simple best-in greedy search for conjunctive query.

    Search starts from the trivial (empty) query and adds an objective-maximizing conditions until no
    further improvement with a single augmentation is possible.

    >>> table = [[1, 1, 1, 1, 0],
    ...          [1, 1, 0, 0, 0],
    ...          [1, 0, 1, 0, 0],
    ...          [0, 1, 1, 1, 1],
    ...          [0, 0, 1, 1, 1],
    ...          [1, 1, 0, 0, 1]]
    >>> ctx = Context.from_tab(table)
    >>> labels = [1, 0, 1, 0, 0, 0]
    >>> from realkd.legacy import impact
    >>> f = impact(labels)
    >>> GreedySearch(ctx, f).run()
    c0 & c2

    """

    def __init__(self, ctx, obj, bdn=None, verbose=False, **kwargs):
        """

        :param Context ctx: the context defining the search space
        :param callable obj: objective function
        :param callable bnd: bounding function satisfying that ``bnd(q) >= max{obj(r) for r in successors(q)}`` (for signature compatibility only, not currently used)
        :param int verbose: level of verbosity

        """
        self.ctx = ctx
        self.f = obj
        self.verbose = verbose

    def run(self):
        """
        Runs the configured search.

        :return: :class:`~realkd.logic.Conjunction` that (approximately) maximizes objective
        """
        intent = SortedSet([])
        extent = self.ctx.extension([])
        value = self.f(extent)
        while True:
            best_i, best_ext = None, None
            for i in range(self.ctx.n):
                if i in intent:
                    continue
                _extent = snp.intersect(extent, self.ctx.extents[i])
                _value = self.f(_extent)
                if _value > value:
                    value = _value
                    best_ext = _extent
                    best_i = i
            if best_i is not None:
                intent.add(best_i)
                extent = best_ext
            else:
                break
            if self.verbose:
                print('*', end='', flush=True)
        return Conjunction(map(lambda i: self.ctx.attributes[i], intent))


class GoldenRatioSearch:
    def __init__(self, func, origin, direction, gradient, epsilon=1e-6) -> None:
        """

        :param func: function to be minimized
        :param origin: origin point
        :param direction: search direction
        :param gradient: the gradient function of func, which is callable
        :param epsilon: the precision of the search
        """
        self.func = func
        self.origin = origin
        self.direction = direction
        self.gradient = gradient
        self.epsilon = epsilon

    def run(self):
        """
        Use golden ratio search to search for an optimal distance along a direction
        to minimize the function

        :return: the distance minimizing the function
        """
        step = (self.origin * self.origin).sum() ** 0.5
        if step == 0.0:
            step = 1.0
        x0 = 0.0
        if self.gradient(self.origin).dot(self.direction) > 0:
            step = -step
        x1 = step
        x = self.origin + x1 * self.direction
        while self.gradient(x).dot(self.direction) * self.gradient(self.origin).dot(self.direction) > 0:
            x1 += step
            x = self.origin + x1 * self.direction
        left = min(x1, x0) - 1.0
        right = max(x1, x0) + 1.0
        ratio = (sqrt(5) - 1) / 2
        diff = right - left
        while right - left > self.epsilon * diff:
            lam = left + (1 - ratio) * (right - left)
            mu = left + ratio * (right - left)
            r_lam = self.func(self.origin + lam * self.direction)
            r_mu = self.func(self.origin + mu * self.direction)
            if r_lam <= r_mu:
                right = mu
            else:
                left = lam
        return (left + right) / 2


#: Dictionary of available search methods.
search_methods = {
    'exhaustive': CoreQueryTreeSearch,
    'greedy': GreedySearch
}

if __name__ == '__main__':
    doctest.testmod()
