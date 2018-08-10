import re
import time
from collections import namedtuple
from functools import partial

import click

from .printers import log_warn, log_info, log_debug
from .utils import closed_range, s2b, parse_raw_assignment_bool, parse_raw_assignment_int
from .solver import StreamSolver, FileSolver

__all__ = ['TruthTable']

VARIABLES = 'nodetype terminal child_left child_right parent value child_value_left child_value_right'


class ParseTree:

    class Node:

        def __init__(self, nodetype, terminal):
            assert 0 <= nodetype <= 4
            self.nodetype = nodetype
            self.terminal = terminal
            self.parent = self.child_left = self.child_right = None

        def eval(self, input_values):
            # input_values :: [1..X]:Bool
            if self.nodetype == 0:  # Terminal
                assert self.terminal != 0
                return input_values[self.terminal]

            elif self.nodetype == 1:  # AND
                assert self.child_left is not None
                assert self.child_right is not None
                return self.child_left.eval(input_values) and self.child_right.eval(input_values)

            elif self.nodetype == 2:  # OR
                assert self.child_left is not None
                assert self.child_right is not None
                return self.child_left.eval(input_values) or self.child_right.eval(input_values)

            elif self.nodetype == 3:  # NOT
                assert self.child_left is not None
                return not self.child_left.eval(input_values)

            elif self.nodetype == 4:  # None
                # return False
                raise ValueError('Maybe think again?')

        def size(self):
            if self.nodetype == 0:
                return 1
            elif self.nodetype == 1 or self.nodetype == 2:
                return 1 + self.child_left.size() + self.child_right.size()
            elif self.nodetype == 3:
                return 1 + self.child_left.size()
            elif self.nodetype == 4:
                # return 0
                raise ValueError('Maybe think again?')

        def __str__(self):
            if self.nodetype == 0:  # Terminal
                return self.names[self.terminal]
            elif self.nodetype == 1:  # AND
                left = str(self.child_left)
                right = str(self.child_right)
                if self.child_left.nodetype == 2:  # Left child is OR
                    left = f'({left})'
                if self.child_right.nodetype == 2:  # Right child is OR
                    right = f'({right})'
                return f'{left} & {right}'
            elif self.nodetype == 2:  # OR
                left = str(self.child_left)
                right = str(self.child_right)
                if self.child_left.nodetype == 1:  # Left child is AND
                    left = f'({left})'
                if self.child_right.nodetype == 1:  # Right child is AND
                    right = f'({right})'
                return f'{left} | {right}'
            elif self.nodetype == 3:  # NOT
                if self.child_left.nodetype == 0:
                    return f'~{self.child_left}'
                else:
                    return f'~({self.child_left})'
            elif self.nodetype == 4:  # None
                raise ValueError(f'why are you trying to display None-typed node?')

    def __init__(self, names, nodetype, terminal, parent, child_left, child_right):
        # Note: all arguments are 1-based
        assert len(nodetype) == len(terminal) == len(parent) == len(child_left) == len(child_right)

        self.Node.names = names
        P = len(nodetype) - 1
        nodes = [None] + [self.Node(nt, tn) for nt, tn in zip(nodetype[1:], terminal[1:])]  # 1-based
        for p in closed_range(1, P):
            nodes[p].parent = nodes[parent[p]]
            nodes[p].child_left = nodes[child_left[p]]
            nodes[p].child_right = nodes[child_right[p]]
        self.root = nodes[1]

    def eval(self, input_values):
        # input_values :: [1..X]:Bool
        return self.root.eval(input_values)

    def size(self):
        return self.root.size()

    def __str__(self):
        return str(self.root)


class TruthTable:

    Reduction = namedtuple('Reduction', VARIABLES)
    Assignment = namedtuple('Assignment', VARIABLES + ' P')

    def __init__(self, names, inputs, values):
        assert len(inputs) == len(values)
        # Note: all variables are 1-based
        self.names = [None] + names
        self.inputs = [None] + [s2b(input_) for input_ in inputs]
        self.values = s2b(values)

    @classmethod
    def from_file(cls, filename):
        with click.open_file(filename) as f:
            inputs = []
            values = []
            names = f.readline().split()

            for line in f:
                m = re.match(r'(?P<lhs>[01]+).+(?P<value>[01])', re.sub(r'\s', '', line))
                if m:
                    lhs, value = m.groups()
                    inputs.append(tuple(lhs))
                    values.append(value)
                else:
                    log_warn(f'Can\'t parse line "{line}"')
        return cls(names, inputs, values)

    @property
    def number_of_variables(self):
        return self.solver.number_of_variables

    @property
    def number_of_clauses(self):
        return self.solver.number_of_clauses

    def infer(self, P, *, solver_cmd):
        self.P = P
        # self.solver = StreamSolver(cmd=solver_cmd)
        self.solver = FileSolver(cmd=solver_cmd, filename_prefix=f'out/tt_P{P}')
        self._declare_reduction()
        raw_assignment = self.solver.solve()

        if raw_assignment:
            assignment = self.parse_raw_assignment(raw_assignment)
            parse_tree = ParseTree(self.names,
                                   assignment.nodetype,
                                   assignment.terminal,
                                   assignment.parent,
                                   assignment.child_left,
                                   assignment.child_right)
            return parse_tree
        else:
            return None

    def _declare_reduction(self):
        P = self.P
        log_debug(f'Declaring reduction for P={P}...')
        time_start_reduction = time.time()

        # =-=-=-=-=-=
        #  CONSTANTS
        # =-=-=-=-=-=

        X = len(self.names) - 1  # 1-based!
        U = len(self.inputs) - 1  # 1-based!

        new_variable = self.solver.new_variable
        add_clause = self.solver.add_clause
        declare_array = self.solver.declare_array
        ALO = self.solver.ALO
        AMO = self.solver.AMO
        imply = self.solver.imply
        iff = self.solver.iff
        iff_and = self.solver.iff_and
        iff_or = self.solver.iff_or

        # =-=-=-=-=-=
        #  VARIABLES
        # =-=-=-=-=-=

        # guards variables
        nodetype = declare_array(P, 3, with_zero=True)
        terminal = declare_array(P, X, with_zero=True)
        parent = declare_array(P, P, with_zero=True)
        child_left = declare_array(P, P, with_zero=True)
        child_right = declare_array(P, P, with_zero=True)
        value = declare_array(P, U)
        child_value_left = declare_array(P, U)
        child_value_right = declare_array(P, U)

        # =-=-=-=-=-=-=
        #  CONSTRAINTS
        # =-=-=-=-=-=-=

        so_far_state = [self.number_of_clauses]

        def so_far():
            now = self.number_of_clauses
            ans = now - so_far_state[0]
            so_far_state[0] = now
            return ans

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # constraints

        # 1. Nodetype constraints
        # 1.0. ALO/AMO(nodetype)
        for p in closed_range(1, P):
            ALO(nodetype[p])
            AMO(nodetype[p])

        # 1.1. AND/OR nodes cannot have numbers P-1 or P
        if P >= 1:
            add_clause(-nodetype[P][1])
            add_clause(-nodetype[P][2])
        if P >= 2:
            add_clause(-nodetype[P - 1][1])
            add_clause(-nodetype[P - 1][2])

        # 1.2. NOT nodes cannot have number P
        add_clause(-nodetype[P][3])

        log_debug(f'1. Clauses: {so_far()}', symbol='STAT')

        # 2. Terminals constraints
        # 2.0. ALO/AMO(terminal)
        for p in closed_range(1, P):
            ALO(terminal[p])
            AMO(terminal[p])

        # 2.1. Only terminals have associated terminal variables
        for p in closed_range(1, P):
            iff(nodetype[p][0], -terminal[p][0])

        # 2.2. Terminals have no children
        for p in closed_range(1, P):
            imply(nodetype[p][0], child_left[p][0])
            imply(nodetype[p][0], child_right[p][0])

        # 2.3. Terminals have value from associated input variable
        for p in closed_range(1, P):
            for x in closed_range(1, X):
                # terminal[p,x] -> AND_u( value[p,u] <-> inputs[u,x] )
                for u in closed_range(1, U):
                    if self.inputs[u][x]:
                        imply(terminal[p][x], value[p][u])
                    else:
                        imply(terminal[p][x], -value[p][u])

        log_debug(f'2. Clauses: {so_far()}', symbol='STAT')

        # 3. Parent and children constraints
        # 3.0. ALO/AMO(parent,child_left,child_right)
        for p in closed_range(1, P):
            ALO(parent[p])
            AMO(parent[p])
        for p in closed_range(1, P):
            ALO(child_left[p])
            AMO(child_left[p])
        for p in closed_range(1, P):
            ALO(child_right[p])
            AMO(child_right[p])

        # 3.1. Root has no parent
        add_clause(parent[1][0])

        # 3.2. BFS: typed nodes (except root) have parent with lesser number
        for p in closed_range(2, P):
            add_clause(-parent[p][0])
            for p_ in closed_range(p + 1, P):
                add_clause(-parent[p][p_])

        # 3.3. parent<->child relation
        for p in closed_range(1, P - 1):
            for ch in closed_range(p + 1, P):
                # parent[ch,p] -> child_left[p,ch] | child_right[p,ch]
                add_clause(-parent[ch][p], child_left[p][ch], child_right[p][ch])

        # 3.4. Node with number P have no children; P-1 -- no right child
        add_clause(child_left[P][0])
        add_clause(child_right[P][0])
        for u in closed_range(1, U):
            add_clause(-child_value_left[P][u])
            add_clause(-child_value_right[P][u])
        if P > 1:
            add_clause(child_right[P - 1][0])
            for u in closed_range(1, U):
                add_clause(-child_value_right[P - 1][u])

        log_debug(f'3. Clauses: {so_far()}', symbol='STAT')

        # 4. AND/OR nodes constraints
        # 4.1. AND/OR: left child has greater number
        for p in closed_range(1, P - 2):
            for p_ in closed_range(0, p):
                imply(nodetype[p][1], -child_left[p][p_])
                imply(nodetype[p][2], -child_left[p][p_])
            imply(nodetype[p][1], -child_left[p][P])
            imply(nodetype[p][2], -child_left[p][P])

        # 4.2. AND/OR: right child is adjacent (+1) to left
        for p in closed_range(1, P - 2):
            for ch in closed_range(p + 1, P - 1):
                # (nodetype[p,1or2] & child_left[p][ch]) -> child_right[p][ch+1]
                for nt in [1, 2]:
                    add_clause(-nodetype[p][nt], -child_left[p][ch], child_right[p][ch + 1])

        # 4.3. AND/OR: children's parents
        for p in closed_range(1, P - 2):
            for ch in closed_range(p + 1, P - 1):
                # (nodetype[p,1or2] & child_left[p,ch]) -> (parent[ch,p] & parent[ch+1,p])
                for nt in [1, 2]:
                    add_clause(-nodetype[p][nt], -child_left[p][ch], parent[ch][p])
                    add_clause(-nodetype[p][nt], -child_left[p][ch], parent[ch + 1][p])

        # 4.4a AND/OR: child_value_left is a value of left child
        for p in closed_range(1, P - 2):
            for ch in closed_range(p + 1, P - 1):
                for u in closed_range(1, U):
                    # (nodetype[p,1or2] & child_left[p,ch]) -> (child_value_left[p,u] <-> value[ch,u])
                    for nt in [1, 2]:
                        add_clause(-nodetype[p][nt], -child_left[p][ch], -child_value_left[p][u], value[ch][u])
                        add_clause(-nodetype[p][nt], -child_left[p][ch], child_value_left[p][u], -value[ch][u])

        # 4.4b AND/OR: child_value_right is a value of right child
        for p in closed_range(1, P - 2):
            for ch in closed_range(p + 2, P):
                for u in closed_range(1, U):
                    # (nodetype[p,1or2] & child_left[p,ch]) -> (child_value_left[p,u] <-> value[ch,u])
                    for nt in [1, 2]:
                        add_clause(-nodetype[p][nt], -child_right[p][ch], -child_value_right[p][u], value[ch][u])
                        add_clause(-nodetype[p][nt], -child_right[p][ch], child_value_right[p][u], -value[ch][u])

        # 4.5a AND: value is calculated as a conjunction of children
        for p in closed_range(1, P - 2):
            for u in closed_range(1, U):
                # nodetype[p,1] -> (value[p,u] <-> child_value_left[p,u] & child_value_right[p,u])
                add_clause(-nodetype[p][1], value[p][u], -child_value_left[p][u], -child_value_right[p][u])
                add_clause(-nodetype[p][1], -value[p][u], child_value_left[p][u])
                add_clause(-nodetype[p][1], -value[p][u], child_value_right[p][u])

        # 4.5b OR: value is calculated as a disjunction of children
        for p in closed_range(1, P - 2):
            for u in closed_range(1, U):
                # nodetype[p,2] -> (value[p,u] <-> child_value_left[p,u] & child_value_right[p,u])
                add_clause(-nodetype[p][2], -value[p][u], child_value_left[p][u], child_value_right[p][u])
                add_clause(-nodetype[p][2], value[p][u], -child_value_left[p][u])
                add_clause(-nodetype[p][2], value[p][u], -child_value_right[p][u])

        log_debug(f'4. Clauses: {so_far()}', symbol='STAT')

        # 5. NOT nodes constraints
        # 5.1. NOT: left child has greater number
        for p in closed_range(1, P - 1):
            for p_ in closed_range(0, p):
                imply(nodetype[p][3], -child_left[p][p_])

        # 5.2. NOT: no right child
        for p in closed_range(1, P - 1):
            imply(nodetype[p][3], child_right[p][0])

        # 5.3. NOT: child's parents
        for p in closed_range(1, P - 1):
            for ch in closed_range(p + 1, P):
                add_clause(-nodetype[p][3], -child_left[p][ch], parent[ch][p])

        # 5.4a NOT: child_value_left is a value of left child
        for p in closed_range(1, P - 1):
            for ch in closed_range(p + 1, P):
                for u in closed_range(1, U):
                    # (nodetype[p,3] & child_left[p,ch]) -> (child_value_left[p,u] <-> value[ch,u])
                    add_clause(-nodetype[p][3], -child_left[p][ch], -child_value_left[p][u], value[ch][u])
                    add_clause(-nodetype[p][3], -child_left[p][ch], child_value_left[p][u], -value[ch][u])

        # 5.4b NOT: child_value_right is False
        for p in closed_range(1, P - 1):
            for u in closed_range(1, U):
                # nodetype[p,3] -> ~child_value_right[p,u]
                imply(nodetype[p][3], -child_value_right[p][u])

        # 5.5. NOT: value is calculated as a negation of child
        for p in closed_range(1, P - 1):
            for u in closed_range(1, U):
                # nodetype[p,3] -> (value[p,u] <-> ~child_value_left[p,u])
                add_clause(-nodetype[p][3], -value[p][u], -child_value_left[p][u])
                add_clause(-nodetype[p][3], value[p][u], child_value_left[p][u])

        log_debug(f'5. Clauses: {so_far()}', symbol='STAT')

        # 6. Root value
        for u in closed_range(1, U):
            if self.values[u]:
                add_clause(value[1][u])
            else:
                add_clause(-value[1][u])

        log_debug(f'6. Clauses: {so_far()}', symbol='STAT')

        # TODO: 7. Tree constraints
        #       7.1. Edges
        #       constraint E = sum (p in 1..P) (bool2int(parent[p] != 0));
        #       7.2. Vertices
        #       constraint V = P;
        #       7.3. Tree equality
        #       constraint E = V - 1;

        # =-=-=-=-=
        #   FINISH
        # =-=-=-=-=

        self.reduction = self.Reduction(
            nodetype=nodetype,
            terminal=terminal,
            child_left=child_left,
            child_right=child_right,
            parent=parent,
            value=value,
            child_value_left=child_value_left,
            child_value_right=child_value_right,
        )

        log_debug(f'Done declaring base reduction ({self.number_of_variables} variables, {self.number_of_clauses} clauses) in {time.time() - time_start_reduction:.2f} s')

    def parse_raw_assignment(self, raw_assignment):
        if raw_assignment is None:
            return None

        log_debug('Building assignment...')
        time_start_assignment = time.time()

        wrapper_int = partial(parse_raw_assignment_int, raw_assignment)
        wrapper_bool = partial(parse_raw_assignment_bool, raw_assignment)

        assignment = self.Assignment(
            nodetype=wrapper_int(self.reduction.nodetype),
            terminal=wrapper_int(self.reduction.terminal),
            child_left=wrapper_int(self.reduction.child_left),
            child_right=wrapper_int(self.reduction.child_right),
            parent=wrapper_int(self.reduction.parent),
            value=wrapper_bool(self.reduction.value),
            child_value_left=wrapper_bool(self.reduction.child_value_left),
            child_value_right=wrapper_bool(self.reduction.child_value_right),
            P=self.P,
        )

        log_debug(f'assignment = {assignment}')

        log_debug(f'Done building assignment in {time.time() - time_start_assignment:.2f} s')
        return assignment
