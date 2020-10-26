"""
===================
create_expressions.py
===================
Author: Michel Vazirani
Date: 07/06/2019
Create a dataset of logic expression trees
"""


import gc
import itertools
import pickle as pkl
import copy
import numpy as np
import sys
import ast
import random
import math
from collections import Iterable

from sly import Lexer
from sly import Parser
from parsing_logic.scan_parse import *

import networkx as nx

from sympy import *


"""
NOTE:

    When adding new operation to any LogicNode, need to:
        1. Make sure the operation id (some integer) is 1 + max(all other operation ids)
        2. Add operation id to the __init__ method of the class
        3. Add operation to the  __init__ method of the LogicTree Class
        4. Update cur_max_op_id in LogicTree __init__ method
        5. Add the operation id to the assortment of operations in LogicTree __init__ (e.g. IDENTITY)

"""




class LogicNode(object):

    def __init__(self, parent):
        self.parent = parent
        self.most_recent_op = -1

    def __str__(self):
        return self.parse()

    def __hash__(self):
        return hash(self.parse())

    def __eq__(self, other):
        return self.parse() == other.parse()

    def set_parent(self, parent):
        self.parent = parent

    def set_most_recent_op(self, op):
        self.most_recent_op = op

    def parse(self):
        def parse_helper(node):
            if isinstance(node, LeafNode):
                return node.token
            elif isinstance(node, UnaryNode):
                # if isinstance(node.arg, LeafNode):
                #     return node.token + parse_helper(node.arg)
                # else:
                #     return node.token + "(" + parse_helper(node.arg) + ")"
                if isinstance(node.arg, UnaryNode):
                    return node.token + "(" + parse_helper(node.arg) + ")"
                else:
                    return node.token + parse_helper(node.arg)
            elif isinstance(node, BinaryNode):
                return "(" + parse_helper(node.left) + node.token + parse_helper(node.right) + ")"
            elif isinstance(node, N_aryNode):
                res = ["("]
                for op in node.operands:
                    res.append(parse_helper(op))
                    res.append(node.token)
                res[-1] = ")"
                return "".join(res)
            else:
                print(type(node))
                print(node.token)
                print(node)
                sys.exit()
        return parse_helper(self)

    def deep_parse(self):

        def parse_helper(node):
            if isinstance(node, LeafNode):
                return [node.token, '('+node.token+')']
            elif isinstance(node, UnaryNode):
                if isinstance(node.arg, LeafNode) or \
                            isinstance(node.arg, UnaryNode):
                    child_strs = parse_helper(node.arg)
                    rets = []
                    for str in child_strs:
                        rets.append(node.token+str)
                        rets.append(node.token+"("+str+")")
                    return rets
                else:
                    child_strs = parse_helper(node.arg)
                    rets = []
                    for str in child_strs:
                        rets.append(node.token+"("+str+")")
                    return rets
            elif isinstance(node, BinaryNode):
                left_child_strs = parse_helper(node.left)
                right_child_strs = parse_helper(node.right)
                rets = []
                for l_str in left_child_strs:
                    for r_str in right_child_strs:
                        rets.append("("+l_str+node.token+r_str+")")
                return rets
            elif isinstance(node, N_aryNode):

                all_op_strs = []
                for op in node.operands:
                    all_op_strs.append(parse_helper(op))
                ret_toks = [[] for _ in range(len(node.operands))]
                for i in range(len(all_op_strs)):
                    op_strs = all_op_strs[i]
                    if i == 0:
                        ret_toks[i] = ['('+str+node.token for str in op_strs]
                    else:
                        ret_toks[i] = []
                        for prev in ret_toks[i-1]:
                            ret_toks[i].extend([prev+str+node.token for str in op_strs])

                rets = []
                for i in range(len(ret_toks[-1])):
                    tmp_str = ret_toks[-1][i][:-1]
                    tmp_str += ')'
                    rets.append(''.join(tmp_str))
                return rets
            else:
                print(type(node))
                print(node.token)
                print(node)
                sys.exit()

        return parse_helper(self)

class LeafNode(LogicNode):

    def __init__(self, parent):
        LogicNode.__init__(self, parent)

class UnaryNode(LogicNode):
    def __init__(self, parent, arg=None):
        LogicNode.__init__(self, parent)
        self.arg = arg

    def set_arg(self, ar):
        ar.set_parent(self)
        self.arg = ar

class BinaryNode(LogicNode):
    def __init__(self, parent, left=None, right=None):
        LogicNode.__init__(self, parent)
        self.left = left
        self.right = right

    def set_lr(self, l, r):
        l.set_parent(self)
        r.set_parent(self)
        self.left = l
        self.right = r

class N_aryNode(LogicNode):
    def __init__(self, parent, operands=None):
        LogicNode.__init__(self, parent)
        self.operands = operands

    def set_operands(self, ops):
        for op in ops:
            op.set_parent(self)
        self.operands = ops






class TrueNode(LeafNode):

    def __init__(self, parent):
        LeafNode.__init__(self, parent)
        self.token = "T"
        self.nodeOps = set([1,2,3,4,5,6,7,8,9,10,51,65,66,67,68])

        self.node_ops_dict = {1:self.new_var_p,\
                                2:self.new_var_q,\
                                3:self.new_var_r,\
                                65:self.new_var_s,\
                                4:self.new_var_notp,\
                                5:self.new_var_notq,\
                                6:self.new_var_notr,\
                                66:self.new_var_nots,\
                                7:self.new_tautology_p,\
                                8:self.new_tautology_q,\
                                9:self.new_tautology_r,\
                                67:self.new_tautology_s,\
                                51:self.neg_t,\
                                10:self.identity,\
                                68:self.identity2\
                                }


    def copy(self):
        t = TrueNode(self.parent)
        t.set_most_recent_op(self.most_recent_op)
        return t

    def neg_t(self):
        notnode = NotNode(self.parent)
        falsenode = FalseNode(notnode)
        notnode.set_arg(falsenode)
        return notnode

    def new_var_p(self):
        ornode = OrNode(self.parent)
        pnode = PNode(ornode)
        ornode.set_operands([self, pnode])
        return ornode

    def new_var_q(self):
        ornode = OrNode(self.parent)
        pnode = QNode(ornode)
        ornode.set_operands([self, pnode])
        return ornode

    def new_var_r(self):
        ornode = OrNode(self.parent)
        pnode = RNode(ornode)
        ornode.set_operands([self, pnode])
        return ornode

    def new_var_s(self):
        ornode = OrNode(self.parent)
        snode = SNode(ornode)
        ornode.set_operands([self, snode])
        return ornode


    def new_var_notp(self):
        ornode = OrNode(self.parent)
        notnode = NotNode(ornode)
        pnode = PNode(notnode)
        notnode.set_arg(pnode)
        ornode.set_operands([self, notnode])
        return ornode

    def new_var_notq(self):
        ornode = OrNode(self.parent)
        notnode = NotNode(ornode)
        qnode = QNode(notnode)
        notnode.set_arg(qnode)
        ornode.set_operands([self, notnode])
        return ornode

    def new_var_notr(self):
        ornode = OrNode(self.parent)
        notnode = NotNode(ornode)
        rnode = RNode(notnode)
        notnode.set_arg(rnode)
        ornode.set_operands([self, notnode])
        return ornode

    def new_var_nots(self):
        ornode = OrNode(self.parent)
        notnode = NotNode(ornode)
        snode = RNode(notnode)
        notnode.set_arg(snode)
        ornode.set_operands([self, notnode])
        return ornode

    def new_tautology_p(self):
        ornode = OrNode(self.parent)
        notnode = NotNode(ornode)
        pnode1 = PNode(notnode)
        notnode.set_arg(pnode1)
        pnode2 = PNode(ornode)
        ornode.set_operands([notnode, pnode2])
        return ornode

    def new_tautology_q(self):
        ornode = OrNode(self.parent)
        notnode = NotNode(ornode)
        qnode1 = QNode(notnode)
        notnode.set_arg(qnode1)
        qnode2 = QNode(ornode)
        ornode.set_operands([notnode, qnode2])
        return ornode

    def new_tautology_r(self):
        ornode = OrNode(self.parent)
        notnode = NotNode(ornode)
        rnode1 = RNode(notnode)
        notnode.set_arg(rnode1)
        rnode2 = RNode(ornode)
        ornode.set_operands([notnode, rnode2])
        return ornode

    def new_tautology_s(self):
        ornode = OrNode(self.parent)
        notnode = NotNode(ornode)
        snode1 = SNode(notnode)
        notnode.set_arg(snode1)
        snode2 = SNode(ornode)
        ornode.set_operands([notnode, snode2])
        return ornode


    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_operands([self, truenode])
        return andnode

    def identity2(self):
        ornode = OrNode(self.parent)
        truenode = TrueNode(ornode)
        ornode.set_operands([self, truenode])
        return ornode


    def do_ops(self, allowed_ops, op_pair_dict):
        todo_ops = self.nodeOps.intersection(allowed_ops)
        if not self.most_recent_op == -1:
            if self.most_recent_op in op_pair_dict:
                todo_ops -= set([op_pair_dict[self.most_recent_op]])
        new_nodes = []
        for op in todo_ops:
            new_node = self.node_ops_dict[op]()
            if new_node:
                new_node.set_most_recent_op(op)
                new_nodes.append((new_node, op))


        new_nodes = [node for node in new_nodes if node[0] is not False]
        return new_nodes


class FalseNode(LeafNode):

    def __init__(self, parent):
        LeafNode.__init__(self, parent)
        self.token = "F"
        self.nodeOps = set([31,32,33,34,35,36,37,38,39,40,11,74,75,76])
        self.node_ops_dict = {31:self.new_var_p_f,\
                                32:self.new_var_q_f,\
                                33:self.new_var_r_f,\
                                74:self.new_var_s_f,\
                                34:self.new_var_notp_f,\
                                35:self.new_var_notq_f,\
                                36:self.new_var_notr_f,\
                                75:self.new_var_nots_f,\
                                37:self.new_fallacy_p,\
                                38:self.new_fallacy_q,\
                                39:self.new_fallacy_r,\
                                76:self.new_fallacy_s,\
                                40:self.neg_f,\
                                11:self.identity\
                                }

    def copy(self):
        f = FalseNode(self.parent)
        f.set_most_recent_op(self.most_recent_op)
        return f

    def neg_f(self):
        notnode = NotNode(self.parent)
        truenode = TrueNode(notnode)
        notnode.set_arg(truenode)
        return notnode

    def new_var_p_f(self):
        andnode = AndNode(self.parent)
        pnode = PNode(andnode)
        andnode.set_operands([self, pnode])
        return andnode

    def new_var_q_f(self):
        andnode = AndNode(self.parent)
        qnode = QNode(andnode)
        andnode.set_operands([self, qnode])
        return andnode

    def new_var_r_f(self):
        andnode = AndNode(self.parent)
        rnode = RNode(andnode)
        andnode.set_operands([self, rnode])
        return andnode

    def new_var_s_f(self):
        andnode = AndNode(self.parent)
        snode = SNode(andnode)
        andnode.set_operands([self, snode])
        return andnode

    def new_var_notp_f(self):
        andnode = AndNode(self.parent)
        notnode = NotNode(andnode)
        pnode = PNode(notnode)
        notnode.set_arg(pnode)
        andnode.set_operands([self, notnode])
        return andnode

    def new_var_notq_f(self):
        andnode = AndNode(self.parent)
        notnode = NotNode(andnode)
        qnode = QNode(notnode)
        notnode.set_arg(qnode)
        andnode.set_operands([self, notnode])
        return andnode

    def new_var_notr_f(self):
        andnode = AndNode(self.parent)
        notnode = NotNode(andnode)
        rnode = RNode(notnode)
        notnode.set_arg(rnode)
        andnode.set_operands([self, notnode])
        return andnode

    def new_var_nots_f(self):
        andnode = AndNode(self.parent)
        notnode = NotNode(andnode)
        snode = SNode(notnode)
        notnode.set_arg(snode)
        andnode.set_operands([self, notnode])
        return andnode

    def new_fallacy_p(self):
        andnode = AndNode(self.parent)
        notnode = NotNode(andnode)
        pnode1 = PNode(notnode)
        notnode.set_arg(pnode1)
        pnode2 = PNode(andnode)
        andnode.set_operands([notnode, pnode2])
        return andnode

    def new_fallacy_q(self):
        andnode = AndNode(self.parent)
        notnode = NotNode(andnode)
        qnode1 = QNode(notnode)
        notnode.set_arg(qnode1)
        qnode2 = QNode(andnode)
        andnode.set_operands([notnode, qnode2])
        return andnode

    def new_fallacy_r(self):
        andnode = AndNode(self.parent)
        notnode = NotNode(andnode)
        rnode1 = RNode(notnode)
        notnode.set_arg(rnode1)
        rnode2 = RNode(andnode)
        andnode.set_operands([notnode, rnode2])
        return andnode

    def new_fallacy_s(self):
        andnode = AndNode(self.parent)
        notnode = NotNode(andnode)
        snode1 = SNode(notnode)
        notnode.set_arg(snode1)
        snode2 = SNode(andnode)
        andnode.set_operands([notnode, snode2])
        return andnode

    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_operands([self, truenode])
        return andnode

    def do_ops(self, allowed_ops, op_pair_dict):

        todo_ops = self.nodeOps.intersection(allowed_ops)
        if not self.most_recent_op == -1:
            if self.most_recent_op in op_pair_dict:
                todo_ops -= set([op_pair_dict[self.most_recent_op]])
        new_nodes = []
        for op in todo_ops:
            new_node = self.node_ops_dict[op]()
            if new_node:
                new_node.set_most_recent_op(op)
                new_nodes.append((new_node, op))



        new_nodes = [node for node in new_nodes if node[0] is not False]
        return new_nodes


class PNode(LeafNode):

    def __init__(self, parent):
        LeafNode.__init__(self, parent)
        self.token = "p"
        self.nodeOps = set([12,45])
        self.node_ops_dict = {12:self.identity,\
                                45:self.identity2\
                                }


    def copy(self):
        p = PNode(self.parent)
        p.set_most_recent_op(self.most_recent_op)
        return p

    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_operands([self.copy(), truenode])
        return andnode

    def identity2(self):
        ornode = OrNode(self.parent)
        pnode = PNode(ornode)
        ornode.set_operands([self.copy(), pnode])
        return ornode


    def do_ops(self, allowed_ops, op_pair_dict):

        todo_ops = self.nodeOps.intersection(allowed_ops)
        if not self.most_recent_op == -1:
            if self.most_recent_op in op_pair_dict:
                todo_ops -= set([op_pair_dict[self.most_recent_op]])
        new_nodes = []
        for op in todo_ops:
            new_node = self.node_ops_dict[op]()
            if new_node:
                new_node.set_most_recent_op(op)
                new_nodes.append((new_node, op))


        # new_nodes = [(self.identity(), 12), \
        #             (self.identity2(), 45)]
        new_nodes = [node for node in new_nodes if node[0] is not False]
        return new_nodes


class QNode(LeafNode):

    def __init__(self, parent):
        LeafNode.__init__(self, parent)
        self.token = "q"
        self.nodeOps = set([13,46])
        self.node_ops_dict = {13:self.identity,\
                                46:self.identity2\
                                }

    def copy(self):
        q = QNode(self.parent)
        q.set_most_recent_op(self.most_recent_op)
        return q


    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_operands([self.copy(), truenode])
        return andnode

    def identity2(self):
        ornode = OrNode(self.parent)
        qnode = QNode(ornode)
        ornode.set_operands([self.copy(), qnode])
        return ornode


    def do_ops(self, allowed_ops, op_pair_dict):

        todo_ops = self.nodeOps.intersection(allowed_ops)
        if not self.most_recent_op == -1:
            if self.most_recent_op in op_pair_dict:
                todo_ops -= set([op_pair_dict[self.most_recent_op]])
        new_nodes = []
        for op in todo_ops:
            new_node = self.node_ops_dict[op]()
            if new_node:
                new_node.set_most_recent_op(op)
                new_nodes.append((new_node, op))

        new_nodes = [node for node in new_nodes if node[0] is not False]
        return new_nodes


class RNode(LeafNode):

    def __init__(self, parent):
        LeafNode.__init__(self, parent)
        self.token = "r"
        self.nodeOps = set([14,47])
        self.node_ops_dict = {14:self.identity,\
                                47:self.identity2\
                                }

    def copy(self):
        r = RNode(self.parent)
        r.set_most_recent_op(self.most_recent_op)
        return r


    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_operands([self.copy(), truenode])
        return andnode

    def identity2(self):
        ornode = OrNode(self.parent)
        rnode = RNode(ornode)
        ornode.set_operands([self.copy(), rnode])
        return ornode


    def do_ops(self, allowed_ops, op_pair_dict):

        todo_ops = self.nodeOps.intersection(allowed_ops)
        if not self.most_recent_op == -1:
            if self.most_recent_op in op_pair_dict:
                todo_ops -= set([op_pair_dict[self.most_recent_op]])

        new_nodes = []
        for op in todo_ops:
            new_node = self.node_ops_dict[op]()
            if new_node:
                new_node.set_most_recent_op(op)
                new_nodes.append((new_node, op))

        new_nodes = [node for node in new_nodes if node[0] is not False]
        return new_nodes


class SNode(LeafNode):

    def __init__(self, parent):
        LeafNode.__init__(self, parent)
        self.token = "s"
        self.nodeOps = set([69,70])
        self.node_ops_dict = {69:self.identity,\
                                70:self.identity2\
                                }


    def copy(self):
        s = SNode(self.parent)
        s.set_most_recent_op(self.most_recent_op)
        return s

    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_operands([self.copy(), truenode])
        return andnode

    def identity2(self):
        ornode = OrNode(self.parent)
        snode = SNode(ornode)
        ornode.set_operands([self.copy(), snode])
        return ornode


    def do_ops(self, allowed_ops, op_pair_dict):

        todo_ops = self.nodeOps.intersection(allowed_ops)
        if not self.most_recent_op == -1:
            if self.most_recent_op in op_pair_dict:
                todo_ops -= set([op_pair_dict[self.most_recent_op]])
        new_nodes = []
        for op in todo_ops:
            new_node = self.node_ops_dict[op]()
            if new_node:
                new_node.set_most_recent_op(op)
                new_nodes.append((new_node, op))


        # new_nodes = [(self.identity(), 12), \
        #             (self.identity2(), 45)]
        new_nodes = [node for node in new_nodes if node[0] is not False]
        return new_nodes




class AndNode(N_aryNode):

    def __init__(self, parent, operands=None):
        N_aryNode.__init__(self, parent, operands)
        self.token = "∧"
        self.nodeOps = set([15,16,17,18,19,41,42,54,56,60,59,64,71,73])
        # complex ops: the return value is a list of replacement nodes
        self.complexOps = set([15,42,63])
        self.node_ops_dict = {15:self.commutative1,
                                16:self.associative3,
                                17:self.associative4,
                                18:self.demorgan4,
                                41:self.flatten1,
                                42:self.expand1,
                                54:self.distribute,
                                56:self.factor,
                                60:self.indempotence,
                                59:self.reduce_identity,
                                63:self.to_iff,
                                64:self.absorption,
                                71:self.domination,
                                73:self.remove_true,
                                19:self.identity}


    def copy(self):
        andnode = AndNode(self.parent)
        andnode.set_most_recent_op(self.most_recent_op)
        andnode.set_operands([op.copy() for op in self.operands])
        return andnode

    def commutative1(self):

        idxs = [i for i in range(len(self.operands))]
        op_dict = {i:self.operands[i] for i in range(len(self.operands))}
        op_perms = []
        for perm in itertools.permutations(idxs):
            if not list(perm) == [i for i in range(len(self.operands))]:
                op_perms.append([op_dict[p].copy() for p in perm])

        # op_perms = [list(p) for p in itertools.permutations(self.operands)]
        # op_perms = [[op.copy() for op in perm] for perm in op_perms]

        new_ands = []

        for perm in op_perms:
            andnode = AndNode(self.parent)
            andnode.set_operands(perm)
            new_ands.append(andnode)

        return new_ands

    def associative3(self):
        if len(self.operands) == 2 and isinstance(self.operands[1], AndNode):
            topandnode = AndNode(self.parent)
            bottomandnode = AndNode(topandnode)
            bottomandnode.set_operands([self.operands[0].copy(), self.operands[1].operands[0].copy()])
            topandnode.set_operands([bottomandnode, self.operands[1].operands[1].copy()])
            return topandnode

        else:
            return False

    def associative4(self):
        if len(self.operands) == 2 and isinstance(self.operands[0], AndNode):
            topandnode = AndNode(self.parent)
            bottomandnode = AndNode(topandnode)
            bottomandnode.set_operands([self.operands[0].operands[1].copy(), self.operands[1].copy()])
            topandnode.set_operands([self.operands[0].operands[0].copy(), bottomandnode])
            return topandnode
        else:
            return False

    def flatten1(self):
        def helper(andnode):
            non_ands = []
            ands = []
            for op in andnode.operands:
                if isinstance(op, AndNode):
                    ands.append(op)
                else:
                    non_ands.append(op.copy())
            if ands:
                for an in ands:
                    non_ands.extend(helper(an))
            return non_ands


        base_operands = []
        has_ands = False
        for op in self.operands:
            if isinstance(op, AndNode):
                base_operands.extend(helper(op))
                has_ands = True
            else:
                base_operands.append(op.copy())

        if not has_ands:
            return False

        new_and_node = AndNode(self.parent)
        new_and_node.set_operands(base_operands)
        return new_and_node

    def expand1(self):

        def op_groups(operands):
            if len(operands) <= 1:
                return []

            groups = []
            for split in range(1,len(operands)):
                left = operands[:split]
                right = operands[split:]

                if len(left) == 1 and len(right) == 1:
                    groups.append(left+right)
                elif len(left) == 1:
                    groups.append([left[0],right])
                elif len(right) == 1:
                    groups.append([left,right[0]])
                else:
                    groups.append([left,right])

                if len(left) > 1 and len(right) > 1:
                    left_groups = op_groups(left)
                    right_groups = op_groups(right)

                    lunduped = []
                    for l in left_groups:
                        if l not in lunduped:
                            lunduped.append(l)
                    runduped = []
                    for r in right_groups:
                        if r not in runduped:
                            runduped.append(r)

                    for l in lunduped:
                        for r in runduped:
                            groups.append([l,r])
                elif len(left) == 1 and len(right) == 1:
                    # Don't want to recurse on only two operands
                    pass
                elif len(left) == 1:
                    right_groups = op_groups(right)
                    unduped = []
                    for r in right_groups:
                        if r not in unduped:
                            unduped.append(r)
                    for r in unduped:
                        groups.append([left[0],r])
                elif len(right) == 1:
                    left_groups = op_groups(left)
                    unduped = []
                    for l in left_groups:
                        if l not in unduped:
                            unduped.append(l)
                    for l in unduped:
                        groups.append([l,right[0]])

            unduped = []
            for g in groups:
                if g not in unduped:
                    unduped.append(g)

            return unduped


        inds = [i for i in range(len(self.operands))]
        groups = op_groups(inds)

        def to_and(inds):
            new_and = AndNode(None)
            new_ops = []
            for i in inds:
                if type(i) == list:
                    new_op = to_and(i)
                    new_op.set_parent(new_and)
                    new_ops.append(new_op)
                else:
                    new_ops.append(self.operands[i].copy())
            new_and.set_operands(new_ops)
            return new_and

        ret_ands = [to_and(g) for g in groups]
        for i in range(len(ret_ands)):
            ret_ands[i].set_parent(self.parent)

        if len(self.operands) > 2:
            for coupling in range(2, len(self.operands)-1):
                for idx in range(len(self.operands)-coupling+1):
                    group = [o.copy() for o in self.operands[idx:idx+coupling]]
                    insert_and = AndNode(self)
                    insert_and.set_operands(group)
                    new_ops = [o.copy() for o in self.operands]
                    for _ in range(coupling):
                        new_ops.pop(idx)
                    new_ops.insert(idx,insert_and)
                    ret_and = AndNode(self.parent)
                    ret_and.set_operands(new_ops)
                    ret_ands.append(ret_and)

        return ret_ands


        if len(self.operands) < 3:
            return False

        for op in self.operands:
            if isinstance(op, AndNode):
                return False

        ret_ands = []
        for size in range(2, len(self.operands)):
            for id in range((len(self.operands)-size)+1):
                group = [op.copy() for op in self.operands[id:id+size]]
                rest = [op.copy() for op in self.operands[:id]]
                rest.extend([op.copy() for op in self.operands[id+size:]])

                new_and_node = AndNode(self.parent)
                group_and_node = AndNode(new_and_node)
                group_and_node.set_operands(group)
                rest.insert(id, group_and_node)
                new_and_node.set_operands(rest)

                ret_ands.append(new_and_node)

        return ret_ands

    def demorgan4(self):

        # for i in range(len(self.operands)-1):
        #     if isinstance(self.operands[i], NotNode) and \
        #             isinstance(self.operands[i+1], NotNode):
        #         l = self.operands[i].arg.copy()
        #         r = self.operands[i+1].arg.copy()
        #         notnode = NotNode(self.parent)
        #         ornode = OrNode(notnode)
        #         ornode.set_operands([l, r])
        #         notnode.set_arg(ornode)
        #
        #         if len(self.operands) == 2:
        #             return notnode
        #         else:
        #             ret_and = AndNode(self.parent)
        #             notnode.set_parent(ret_and)
        #             new_ops = [o.copy() for o in self.operands]
        #             new_ops.pop(i)
        #             new_ops[i] = notnode
        #             ret_and.set_operands(new_ops)
        #             return ret_and
        # else:
        #     return False

        retnot = NotNode(self.parent)
        ornode = OrNode(retnot)
        new_ops = [o.copy() for o in self.operands]
        for i in range(len(new_ops)):
            notnode = NotNode(ornode)
            notnode.set_arg(new_ops[i])
            new_ops[i] = notnode
        ornode.set_operands(new_ops)
        retnot.set_arg(ornode)
        return retnot

    def absorption(self):
        # NOTE, CAN'T ABSORB AN OR OPERATION INTO ANOTHER
        # WITH THIS IMPLEMENTATION

        oridxs = []
        nonoridxs = []
        for i in range(len(self.operands)-1, -1, -1):
            if isinstance(self.operands[i], OrNode):
                oridxs.append(i)
            else:
                nonoridxs.append(i)
        if not oridxs:
            return False

        new_ops = [o.copy() for o in self.operands]

        for nonidx in nonoridxs:
            for oridx in oridxs:
                if new_ops[nonidx] in new_ops[oridx].operands:
                    new_ops.pop(nonidx)

        retand = AndNode(self.parent)
        retand.set_operands(new_ops)
        return retand

    def distribute(self):
        # 𝑝 ∧ (𝑞 ∨ 𝑟) ≡ (𝑝 ∧ 𝑞) ∨ (𝑝 ∧ 𝑟)

        for i in range(len(self.operands)-1):

            # if isinstance(self.operands[i], LeafNode) and\
            if isinstance(self.operands[i+1], OrNode) and\
                len(self.operands[i+1].operands) == 2:
                ornode = OrNode(self.parent)
                leftand = AndNode(ornode)
                rightand = AndNode(ornode)
                left_args = [self.operands[i].copy(), self.operands[i+1].operands[0].copy()]
                right_args = [self.operands[i].copy(), self.operands[i+1].operands[1].copy()]
                leftand.set_operands(left_args)
                rightand.set_operands(right_args)
                ornode.set_operands([leftand, rightand])

                if len(self.operands) == 2:
                    return ornode
                else:
                    ret_and = AndNode(self.parent)
                    ornode.set_parent(ret_and)
                    new_operands = [op.copy() for op in self.operands]
                    new_operands.pop(i)
                    new_operands.pop(i)
                    new_operands.insert(i, ornode)
                    ret_and.set_operands(new_operands)

                    return ret_and


            # elif isinstance(self.operands[i+1], LeafNode) and\
            elif isinstance(self.operands[i], OrNode) and\
                len(self.operands[i].operands) == 2:
                ornode = OrNode(self.parent)
                leftand = AndNode(ornode)
                rightand = AndNode(ornode)
                right_args = [self.operands[i+1].copy(), self.operands[i].operands[0].copy()]
                left_args = [self.operands[i+1].copy(), self.operands[i].operands[1].copy()]
                rightand.set_operands(right_args)
                leftand.set_operands(left_args)
                ornode.set_operands([leftand, rightand])

                if len(self.operands) == 2:
                    return ornode
                else:
                    ret_and = AndNode(self.parent)
                    ornode.set_parent(ret_and)
                    new_operands = [op.copy() for op in self.operands]
                    new_operands.pop(i)
                    new_operands.pop(i)
                    new_operands.insert(i, ornode)
                    ret_and.set_operands(new_operands)

                    return ret_and

        else:
            return False

    def factor(self):
        # 𝑝 ∨ (𝑞 ∧𝑟) ≡ (𝑝 ∨𝑞) ∧ (𝑝 ∨𝑟)

        for i in range(len(self.operands)-1):
            if isinstance(self.operands[i], OrNode) and\
                isinstance(self.operands[i+1], OrNode) and\
                len(self.operands[i].operands) == len(self.operands[i+1].operands) == 2:

                # Check if the or nodes have a common operand
                all_oprnds = self.operands[i].operands + self.operands[i+1].operands
                oprnd_counts = {}
                for op in all_oprnds:
                    if op in oprnd_counts:
                        oprnd_counts[op] += 1
                    else:
                        oprnd_counts[op] = 1
                dup_op = None
                unq_ops = []
                for op in all_oprnds:
                    if oprnd_counts[op] == 2:
                        dup_op = op.copy()
                    else:
                        unq_ops.append(op.copy())

                if dup_op and len(unq_ops) == 2:
                    if len(self.operands) > 2:
                        ret_and = AndNode(self.parent)
                        ornode = OrNode(ret_and)
                        andnode = AndNode(ornode)
                        andnode.set_operands(unq_ops)
                        ornode.set_operands([dup_op, andnode])

                        new_operands = [op.copy() for op in self.operands]
                        new_operands.pop(i)
                        new_operands.pop(i)
                        new_operands.insert(i, ornode)
                        ret_and.set_operands(new_operands)

                        return ret_and

                    else:
                        ornode = OrNode(self.parent)
                        andnode = AndNode(ornode)
                        andnode.set_operands(unq_ops)
                        ornode.set_operands([dup_op, andnode])
                        return ornode

                else:
                    return False
            else:
                return False

    def indempotence(self):
        if len(self.operands) == 2 and self.operands[0].parse() == self.operands[1].parse():
            new_node = self.operands[0].copy()
            new_node.set_parent(self.parent)
            return new_node
        else:
            for i in range(len(self.operands)-1):
                if self.operands[i].parse() == self.operands[i+1].parse():
                    new_operands = [op.copy() for op in self.operands]
                    dup_op = new_operands.pop(i)
                    new_operands[i] = dup_op
                    andnode = AndNode(self.parent)
                    andnode.set_operands(new_operands)

                    return andnode
            return False

    def reduce_identity(self):
        for i in range(len(self.operands)):
            if isinstance(self.operands[i], TrueNode):
                if len(self.operands) == 2:
                    other_op = self.operands[1-i]
                    return other_op.copy()
                else:
                    new_ops = [o.copy() for o in self.operands]
                    new_ops.pop(i)
                    ret_and = AndNode(self.parent)
                    ret_and.set_operands(new_ops)
                    return ret_and
        else:
            return False

    def negation(self):
        # need to decide whether to return just false or insert false in the operands
        for i in range(len(self.operands)-1):
            if isinstance(self.operands[i], NotNode):
                tmp_node = self.operands[i].arg.copy()
                if tmp_node == self.operands[i+1]:
                    if len(self.operands) > 2:
                        ret_and = AndNode(self.parent)
                        new_operands = [op.copy() for op in self.operands]
                        new_operands.pop(i)
                        new_operands.pop(i)
                        new_operands.insert(i, FalseNode(self.parent))
                        ret_and.set_operands(new_operands)
                        return ret_and

                    else:
                        return FalseNode(self.parent)
            elif isinstance(self.operands[i+1], NotNode):
                tmp_node = self.operands[i+1].arg.copy()
                if tmp_node == self.operands[i]:
                    if len(self.operands) > 2:
                        ret_and = AndNode(self.parent)
                        new_operands = [op.copy() for op in self.operands]
                        new_operands.pop(i)
                        new_operands.pop(i)
                        new_operands.insert(i, FalseNode(self.parent))
                        ret_and.set_operands(new_operands)
                        return ret_and

                    else:
                        return FalseNode(self.parent)
        else:
            return False

    def domination(self):
        for i in range(len(self.operands)):
            if isinstance(self.operands[i], FalseNode):
                return FalseNode(self.parent)
        else:
            return False

    def remove_true(self):
        true_idxs = []
        for i in range(len(self.operands)-1, -1, -1):
            if isinstance(self.operands[i], TrueNode):
                true_idxs.append(i)
        if not true_idxs:
            return False
        new_ops = [o.copy() for o in self.operands]
        for idx in true_idxs:
            new_ops.pop(idx)
        if len(new_ops) == 1:
            return new_ops[0]
        retor = OrNode(self.parent)
        retor.set_operands(new_ops)
        return retor

    def to_iff(self):
        for i in range(len(self.operands)-1):
            if isinstance(self.operands[i], ImplicationNode) and\
                isinstance(self.operands[i+1], ImplicationNode) and\
                self.operands[i].left == self.operands[i+1].right and\
                self.operands[i].right == self.operands[i+1].left:

                dblimp1 = DblimplicationNode(self.parent)
                op1 = self.operands[i].left.copy()
                op2 = self.operands[i].right.copy()
                dblimp1.set_lr(op1,op2)

                dblimp2 = DblimplicationNode(self.parent)
                dblimp2.set_lr(op2.copy(),op1.copy())

                if len(self.operands) == 2:
                    return [dblimp1, dblimp2]
                else:
                    new_ops1 = [o.copy() for o in self.operands]
                    new_ops1.pop(i)
                    new_ops1[i] = dblimp1
                    ret_and1 = AndNode(self.parent)
                    ret_and1.set_operands(new_ops1)

                    new_ops2 = [o.copy() for o in self.operands]
                    new_ops2.pop(i)
                    new_ops2[i] = dblimp2
                    ret_and2 = AndNode(self.parent)
                    ret_and2.set_operands(new_ops2)

                    return [ret_and1, ret_and2]
        else:
            return False

    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_operands([self, truenode])
        return andnode

    def do_ops(self, allowed_ops, op_pair_dict):
        todo_ops = self.nodeOps.intersection(allowed_ops)
        todo_ops -= self.complexOps
        if not self.most_recent_op == -1:
            if self.most_recent_op in op_pair_dict:
                todo_ops -= set([op_pair_dict[self.most_recent_op]])
                allowed_ops -= set([op_pair_dict[self.most_recent_op]])


        new_nodes = []
        for op in todo_ops:
            if op not in self.complexOps:
                new_node = self.node_ops_dict[op]()
                if new_node:
                    new_node.set_most_recent_op(op)
                    new_nodes.append((new_node, op))

        for op in self.complexOps:
            if op in allowed_ops:
                news = self.node_ops_dict[op]()
                if news:
                    for new_node in news:
                        new_node.set_most_recent_op(op)
                        new_nodes.append((new_node,op))

        new_nodes = [node for node in new_nodes if node[0] is not False]
        return new_nodes





class OrNode(N_aryNode):

    def __init__(self, parent, operands=None):
        N_aryNode.__init__(self, parent, operands)
        self.token = "∨"
        self.nodeOps = set([20,21,22,23,24,25,43,44,48,49,50,55,57,72])
        # complex ops: the return value is a list of replacement nodes
        self.complexOps = set([20,44])
        self.node_ops_dict = {20:self.commutative2,
                                21:self.associative1,
                                22:self.associative2,
                                23:self.logic_equiv,
                                24:self.demorgan3,
                                43:self.flatten2,
                                25:self.identity,
                                48:self.domination,
                                49:self.indempotence,
                                55:self.distribute,
                                57:self.factor,
                                50:self.negation,
                                72:self.remove_false,
                                44:self.expand2}



    def copy(self):
        ornode = OrNode(self.parent)
        ornode.set_most_recent_op(self.most_recent_op)
        ornode.set_operands([op.copy() for op in self.operands])
        return ornode

    def commutative2(self):

        idxs = [i for i in range(len(self.operands))]
        op_dict = {i:self.operands[i] for i in range(len(self.operands))}
        op_perms = []
        for perm in itertools.permutations(idxs):
            if not list(perm) == [i for i in range(len(self.operands))]:
                op_perms.append([op_dict[p].copy() for p in perm])


        # op_perms = [list(p) for p in itertools.permutations(self.operands)]
        # op_perms = [[op.copy() for op in perm] for perm in op_perms]

        new_ors = []

        for perm in op_perms:
            ornode = OrNode(self.parent)
            ornode.set_operands(perm)
            new_ors.append(ornode)

        return new_ors

    def associative1(self):
        if len(self.operands) == 2 and isinstance(self.operands[1], OrNode):
            topornode = OrNode(self.parent)
            bottomornode = OrNode(topornode)
            bottomornode.set_operands([self.operands[0].copy(), self.operands[1].operands[0].copy()])
            right_ops = [op.copy() for op in self.operands[1].operands[1:]]
            topornode.set_operands([bottomornode]+right_ops)
            return topornode
        else:
            return False

    def associative2(self):
        if len(self.operands) == 2 and isinstance(self.operands[0], OrNode):
            topornode = OrNode(self.parent)
            bottomornode = OrNode(topornode)
            bottomornode.set_operands([self.operands[0].operands[-1].copy(), self.operands[1].copy()])
            left_ops = [op.copy() for op in self.operands[0].operands[:-1]]
            topornode.set_operands(left_ops+[bottomornode])
            return topornode
        else:
            return False

    def flatten2(self):
        def helper(ornode):
            non_ors = []
            ors = []
            for op in ornode.operands:
                if isinstance(op, OrNode):
                    ors.append(op)
                else:
                    non_ors.append(op.copy())
            if ors:
                for o in ors:
                    non_ors.extend(helper(o))
            return non_ors


        base_operands = []
        has_ors = False
        for op in self.operands:
            if isinstance(op, OrNode):
                base_operands.extend(helper(op))
                has_ors = True
            else:
                base_operands.append(op.copy())

        if not has_ors:
            return False

        new_or_node = OrNode(self.parent)
        new_or_node.set_operands(base_operands)
        return new_or_node

    def expand2(self):

        def op_groups(operands):
            if len(operands) <= 1:
                return []

            groups = []
            for split in range(1,len(operands)):
                left = operands[:split]
                right = operands[split:]

                if len(left) == 1 and len(right) == 1:
                    groups.append(left+right)
                elif len(left) == 1:
                    groups.append([left[0],right])
                elif len(right) == 1:
                    groups.append([left,right[0]])
                else:
                    groups.append([left,right])

                if len(left) > 1 and len(right) > 1:
                    left_groups = op_groups(left)
                    right_groups = op_groups(right)

                    lunduped = []
                    for l in left_groups:
                        if l not in lunduped:
                            lunduped.append(l)
                    runduped = []
                    for r in right_groups:
                        if r not in runduped:
                            runduped.append(r)

                    for l in lunduped:
                        for r in runduped:
                            groups.append([l,r])
                elif len(left) == 1 and len(right) == 1:
                    # Don't want to recurse on only two operands
                    pass
                elif len(left) == 1:
                    right_groups = op_groups(right)
                    unduped = []
                    for r in right_groups:
                        if r not in unduped:
                            unduped.append(r)
                    for r in unduped:
                        groups.append([left[0],r])
                elif len(right) == 1:
                    left_groups = op_groups(left)
                    unduped = []
                    for l in left_groups:
                        if l not in unduped:
                            unduped.append(l)
                    for l in unduped:
                        groups.append([l,right[0]])

            unduped = []
            for g in groups:
                if g not in unduped:
                    unduped.append(g)

            return unduped


        inds = [i for i in range(len(self.operands))]
        groups = op_groups(inds)

        def to_or(inds):
            new_or = OrNode(None)
            new_ops = []
            for i in inds:
                if type(i) == list:
                    new_op = to_or(i)
                    new_op.set_parent(new_or)
                    new_ops.append(new_op)
                else:
                    new_ops.append(self.operands[i].copy())
            new_or.set_operands(new_ops)
            return new_or

        ret_ors = [to_or(g) for g in groups]
        for i in range(len(ret_ors)):
            ret_ors[i].set_parent(self.parent)

        if len(self.operands) > 2:
            for coupling in range(2, len(self.operands)-1):
                for idx in range(len(self.operands)-coupling+1):
                    group = [o.copy() for o in self.operands[idx:idx+coupling]]
                    insert_or = OrNode(self)
                    insert_or.set_operands(group)
                    new_ops = [o.copy() for o in self.operands]
                    for _ in range(coupling):
                        new_ops.pop(idx)
                    new_ops.insert(idx,insert_or)
                    ret_or = OrNode(self.parent)
                    ret_or.set_operands(new_ops)
                    ret_ors.append(ret_or)

        return ret_ors

        ret_ors = []
        for size in range(2, len(self.operands)):
            for id in range((len(self.operands)-size)+1):
                group = [op.copy() for op in self.operands[id:id+size]]
                rest = [op.copy() for op in self.operands[:id]]
                rest.extend([op.copy() for op in self.operands[id+size:]])

                new_or_node = OrNode(self.parent)
                group_or_node = OrNode(new_or_node)
                group_or_node.set_operands(group)
                rest.insert(id, group_or_node)
                new_or_node.set_operands(rest)

                ret_ors.append(new_or_node)

        return ret_ors

    def logic_equiv(self):
        if (isinstance(self.operands[0], NotNode) and not isinstance(self.operands[1], NotNode)):
            implies = ImplicationNode(self.parent)
            condition = self.operands[0].arg.copy()
            implication = self.operands[1].copy()
            implies.set_lr(condition, implication)
            return implies
        elif (isinstance(self.operands[1], NotNode) and not isinstance(self.operands[0], NotNode)):
            implies = ImplicationNode(self.parent)
            condition = self.operands[1].arg.copy()
            implication = self.operands[0].copy()
            implies.set_lr(condition, implication)
            return implies
        else:
            return False

    def demorgan3(self):
        retnot = NotNode(self.parent)
        andnode = AndNode(retnot)
        new_ops = [o.copy() for o in self.operands]
        for i in range(len(new_ops)):
            notnode = NotNode(andnode)
            notnode.set_arg(new_ops[i])
            new_ops[i] = notnode
        andnode.set_operands(new_ops)
        retnot.set_arg(andnode)
        return retnot



    # def demorg_factor(self):
    #     retnot = NotNode(self.parent)
    #     andnode = AndNode(retnot)
    #     new_ops = [o.copy() for o in self.operands]
    #     for i in range(len(new_ops)):
    #         notnode = NotNode(andnode)
    #         notnode.set_arg(new_ops[i])
    #         new_ops[i] = notnode
    #     andnode.set_operands(new_ops)
    #     return retnot


    def domination(self):
        for i in range(len(self.operands)):
            if isinstance(self.operands[i], TrueNode):
                return TrueNode(self.parent)
        else:
            return False


    def remove_false(self):
        false_idxs = []
        for i in range(len(self.operands)-1, -1, -1):
            if isinstance(self.operands[i], FalseNode):
                false_idxs.append(i)
        if not false_idxs:
            return False
        new_ops = [o.copy() for o in self.operands]
        for idx in false_idxs:
            new_ops.pop(idx)
        if len(new_ops) == 1:
            return new_ops[0]
        retor = OrNode(self.parent)
        retor.set_operands(new_ops)
        return retor

    def indempotence(self):
        if len(self.operands) == 2 and self.operands[0].parse() == self.operands[1].parse():
            new_node = self.operands[0].copy()
            new_node.set_parent(self.parent)
            return new_node
        else:
            for i in range(len(self.operands)-1):
                if self.operands[i].parse() == self.operands[i+1].parse():
                    new_operands = [op.copy() for op in self.operands]
                    dup_op = new_operands.pop(i)
                    new_operands.pop(i)
                    ornode = OrNode(self.parent)
                    new_operands.insert(i, dup_op)
                    ornode.set_operands(new_operands)

                    return ornode
            return False

    def distribute(self):
        # 𝑝 ∨ (𝑞 ∧𝑟) ≡ (𝑝 ∨𝑞) ∧ (𝑝 ∨𝑟)
        # if len(self.operands) == 2: #GOTTA CHANGE THIS TO ITERATE OVER ALL OPERANDS INSTEAD OF JUST WHEN THERE ARE 2

        for i in range(len(self.operands)-1):
            # if isinstance(self.operands[i], LeafNode) and\
            if isinstance(self.operands[i+1], AndNode) and\
                len(self.operands[i+1].operands) == 2:
                andnode = AndNode(self.parent)
                leftor = OrNode(andnode)
                rightor = OrNode(andnode)
                left_args = [self.operands[i].copy(), self.operands[i+1].operands[0].copy()]
                right_args = [self.operands[i].copy(), self.operands[i+1].operands[1].copy()]
                leftor.set_operands(left_args)
                rightor.set_operands(right_args)
                andnode.set_operands([leftor, rightor])

                if len(self.operands) == 2:
                    return andnode
                else:
                    ret_or = OrNode(self.parent)
                    andnode.set_parent(ret_or)
                    new_operands = [op.copy() for op in self.operands]
                    new_operands.pop(i)
                    new_operands.pop(i)
                    new_operands.insert(i, andnode)
                    ret_or.set_operands(new_operands)

                    return ret_or


            # elif isinstance(self.operands[i+1], LeafNode) and\
            elif isinstance(self.operands[i], AndNode) and\
                len(self.operands[i].operands) == 2:
                andnode = AndNode(self.parent)
                leftor = OrNode(andnode)
                rightor = OrNode(andnode)
                right_args = [self.operands[i+1].copy(), self.operands[i].operands[0].copy()]
                left_args = [self.operands[i+1].copy(), self.operands[i].operands[1].copy()]
                rightor.set_operands(right_args)
                leftor.set_operands(left_args)
                andnode.set_operands([leftor, rightor])

                if len(self.operands) == 2:
                    return andnode
                else:
                    ret_or = OrNode(self.parent)
                    andnode.set_parent(ret_or)
                    new_operands = [op.copy() for op in self.operands]
                    new_operands.pop(i)
                    new_operands.pop(i)
                    new_operands.insert(i, andnode)
                    ret_or.set_operands(new_operands)

                    return ret_or

        else:
            return False

    def factor(self):
        # 𝑝 ∧ (𝑞 ∨ 𝑟) ≡ (𝑝 ∧ 𝑞) ∨ (𝑝 ∧ 𝑟)

        for i in range(len(self.operands)-1):
            if isinstance(self.operands[i], AndNode) and\
                isinstance(self.operands[i+1], AndNode) and\
                len(self.operands[i].operands) == len(self.operands[i+1].operands) == 2:
                # all((isinstance(op, LeafNode) or isinstance(op, UnaryNode)) for op in self.operands[i].operands) and\
                # all((isinstance(op, LeafNode) or isinstance(op, UnaryNode)) for op in self.operands[i+1].operands):

                # Check if the or nodes have a common operand
                all_oprnds = self.operands[i].operands + self.operands[i+1].operands
                # all_oprnds = [op.parse() for op in all_oprnds]
                oprnd_counts = {}
                for op in all_oprnds:
                    if op in oprnd_counts:
                        oprnd_counts[op] += 1
                    else:
                        oprnd_counts[op] = 1
                dup_op = None
                unq_ops = []
                for op in all_oprnds:
                    if oprnd_counts[op] == 2:
                        dup_op = op.copy()
                    else:
                        unq_ops.append(op.copy())

                if dup_op and len(unq_ops) == 2:
                    assert(len(unq_ops) == 2)
                    if len(self.operands) > 2:
                        ret_or = OrNode(self.parent)
                        andnode = AndNode(ret_or)
                        ornode = OrNode(andnode)
                        ornode.set_operands(unq_ops)
                        andnode.set_operands([dup_op, ornode])

                        new_operands = [op.copy() for op in self.operands]
                        new_operands.pop(i)
                        new_operands.pop(i)
                        new_operands.insert(i, andnode)
                        ret_or.set_operands(new_operands)
                        return ret_or

                    else:
                        andnode = AndNode(self.parent)
                        ornode = OrNode(andnode)
                        ornode.set_operands(unq_ops)
                        andnode.set_operands([dup_op, ornode])
                        return andnode

                else:
                    return False
            else:
                return False

    def negation(self):
        for i in range(len(self.operands)-1):
            if isinstance(self.operands[i], NotNode):
                tmp_node = self.operands[i].arg.copy()
                if tmp_node == self.operands[i+1]:
                    if len(self.operands) > 2:
                        ret_or = OrNode(self.parent)
                        new_operands = [op.copy() for op in self.operands]
                        new_operands.pop(i)
                        new_operands[i] = TrueNode(ret_or)
                        ret_or.set_operands(new_operands)
                        return ret_or

                    else:
                        return TrueNode(self.parent)
            elif isinstance(self.operands[i+1], NotNode):
                tmp_node = self.operands[i+1].arg.copy()
                if tmp_node == self.operands[i]:
                    if len(self.operands) > 2:
                        ret_or = OrNode(self.parent)
                        new_operands = [op.copy() for op in self.operands]
                        new_operands.pop(i)
                        new_operands[i] = TrueNode(self.parent)
                        ret_or.set_operands(new_operands)
                        return ret_or

                    else:
                        return TrueNode(self.parent)
        else:
            return False

    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_operands([self, truenode])
        return andnode


    def do_ops(self, allowed_ops, op_pair_dict):

        todo_ops = self.nodeOps.intersection(allowed_ops)
        todo_ops -= self.complexOps
        if not self.most_recent_op == -1:
            if self.most_recent_op in op_pair_dict:
                todo_ops -= set([op_pair_dict[self.most_recent_op]])
                allowed_ops -= set([op_pair_dict[self.most_recent_op]])

        new_nodes = []
        for op in todo_ops:
            if op not in self.complexOps:
                new_node = self.node_ops_dict[op]()
                if new_node:
                    new_node.set_most_recent_op(op)
                    new_nodes.append((new_node, op))

        for op in self.complexOps:
            if op in allowed_ops:
                news = self.node_ops_dict[op]()
                if news:
                    for new_node in news:
                        new_node.set_most_recent_op(op)
                        new_nodes.append((new_node,op))

        new_nodes = [node for node in new_nodes if node[0]]
        return new_nodes


class ImplicationNode(BinaryNode):

    def __init__(self, parent, left=None, right=None):
        BinaryNode.__init__(self, parent, left, right)
        self.token = "→"
        self.nodeOps = set([26,27])
        # complex ops: the return value is a list of replacement nodes
        self.complexOps = set([26])
        self.node_ops_dict = {26:self.to_or,\
                                27:self.identity\
                                }


    def copy(self):
        imp = ImplicationNode(self.parent, self.left.copy(), self.right.copy())
        imp.set_most_recent_op(self.most_recent_op)
        return imp

    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_operands([self, truenode])
        return andnode

    def to_or(self):
        ornode = OrNode(self.parent)
        notnode = NotNode(ornode)
        notnode.set_arg(self.left.copy())
        ornode.set_operands([notnode, self.right.copy()])

        ornode2 = OrNode(self.parent)
        ornode2.set_operands([self.right.copy(), notnode.copy()])

        return [ornode, ornode2]


    def do_ops(self, allowed_ops, op_pair_dict):

        todo_ops = self.nodeOps.intersection(allowed_ops)
        todo_ops -= self.complexOps
        if not self.most_recent_op == -1:
            if self.most_recent_op in op_pair_dict:
                todo_ops -= set([op_pair_dict[self.most_recent_op]])
        new_nodes = []
        for op in todo_ops:
            if op not in self.complexOps:
                new_node = self.node_ops_dict[op]()
                if new_node:
                    new_node.set_most_recent_op(op)
                    new_nodes.append((new_node, op))

        for op in self.complexOps:
            if op in allowed_ops:
                news = self.node_ops_dict[op]()
                if news:
                    for new_node in news:
                        new_node.set_most_recent_op(op)
                        new_nodes.append((new_node,op))



        new_nodes = [node for node in new_nodes if node[0] is not False]
        return new_nodes


class DblimplicationNode(BinaryNode):

    def __init__(self, parent, left=None, right=None):
        BinaryNode.__init__(self, parent, left, right)
        self.token = "↔"
        self.nodeOps = set([61,62])
        # complex ops: the return value is a list of replacement nodes
        self.complexOps = set([61])
        self.node_ops_dict = {61:self.to_and,\
                                62:self.identity\
                                }


    def copy(self):
        dblimp = DblimplicationNode(self.parent, self.left.copy(), self.right.copy())
        dblimp.set_most_recent_op(self.most_recent_op)
        return dblimp

    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_operands([self, truenode])
        return andnode

    def to_and(self):
        andnode = AndNode(self.parent)
        leftimp = ImplicationNode(andnode)
        rightimp = ImplicationNode(andnode)
        leftimp.set_lr(self.left.copy(), self.right.copy())
        rightimp.set_lr(self.right.copy(), self.left.copy())
        andnode.set_operands([leftimp, rightimp])

        andnode2 = AndNode(self.parent)
        andnode2.set_operands([rightimp.copy(),leftimp.copy()])

        return [andnode,andnode2]


    def do_ops(self, allowed_ops, op_pair_dict):

        todo_ops = self.nodeOps.intersection(allowed_ops)
        if not self.most_recent_op == -1:
            if self.most_recent_op in op_pair_dict:
                todo_ops -= set([op_pair_dict[self.most_recent_op]])
        new_nodes = []
        for op in todo_ops:
            if op not in self.complexOps:
                new_node = self.node_ops_dict[op]()
                if new_node:
                    new_node.set_most_recent_op(op)
                    new_nodes.append((new_node, op))

        for op in self.complexOps:
            if op in allowed_ops:
                news = self.node_ops_dict[op]()
                if news:
                    for new_node in news:
                        new_node.set_most_recent_op(op)
                        new_nodes.append((new_node,op))

        new_nodes = [node for node in new_nodes if node[0]]
        return new_nodes


class NotNode(UnaryNode):

    def __init__(self, parent, arg=None):
        UnaryNode.__init__(self, parent, arg)
        self.token = "~"
        self.nodeOps = set([28,29,30,52,53,58])
        self.node_ops_dict = {28:self.demorgan1,\
                                29:self.demorgan2,\
                                52:self.negate_f,\
                                53:self.negate_t,\
                                58:self.double_negation,\
                                30:self.identity\
                                }

    def copy(self):
        nt = NotNode(self.parent, self.arg.copy())
        nt.set_most_recent_op(self.most_recent_op)
        # return NotNode(self.parent, self.arg.copy())
        return nt

    def negate_f(self):
        if isinstance(self.arg, FalseNode):
            return TrueNode(self.parent)

    def negate_t(self):
        if isinstance(self.arg, TrueNode):
            return FalseNode(self.parent)

    def demorgan1(self):
        if isinstance(self.arg, AndNode) and len(self.arg.operands) == 2:
            andnode = self.arg
            ornode = OrNode(self.parent)
            ornode.set_operands([NotNode(ornode, andnode.operands[0].copy()),\
                                NotNode(ornode, andnode.operands[1].copy())])
            return ornode
        else:
            return False

    def demorgan2(self):
        if isinstance(self.arg, OrNode) and len(self.arg.operands) == 2:
            ornode = self.arg
            andnode = AndNode(self.parent)
            andnode.set_operands([NotNode(andnode, ornode.operands[0].copy()),\
                                NotNode(andnode, ornode.operands[1].copy())])
            return andnode
        else:
            return False

    def double_negation(self):
        if isinstance(self.arg, NotNode):
            return self.arg.arg.copy()
        else:
            return False

    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_operands([self, truenode])
        return andnode

    def do_ops(self, allowed_ops, op_pair_dict):
        todo_ops = self.nodeOps.intersection(allowed_ops)
        if not self.most_recent_op == -1:
            if self.most_recent_op in op_pair_dict:
                todo_ops -= set([op_pair_dict[self.most_recent_op]])
        new_nodes = []
        for op in todo_ops:
            new_node = self.node_ops_dict[op]()
            if new_node:
                new_node.set_most_recent_op(op)
                new_nodes.append((new_node, op))

        # new_nodes = [(self.demorgan1(), 28), \
        #             (self.demorgan2(), 29), \
        #             (self.negate_f(), 52), \
        #             (self.negate_t(), 53), \
        #             (self.identity(), 30) \
        #             ]

        new_nodes = [node for node in new_nodes if node[0] is not False]
        return new_nodes


class LogicTree():


    def __init__(self, postfix_tree=None, blowup_control=True, op_seq=None, op_pairs=True):
        if postfix_tree != None:
            self.construct_tree(postfix_tree)
            self.set_computed_ops(0)
        else:
            self.root = None
            self.computed_ops = 0

        self.blowup_control=blowup_control

        cur_max_op_id = 76
        self.all_ops = set([i for i in range(1,cur_max_op_id+1)])

        self.IDENTITY = [10,11,12,13,14,19,25,27,62,30,59,68,69,72]
        self.BOOLEAN_EQUIVALENCE = [51,40,52,53]
        self.IMP_TO_DISJ = [23,26,61,63]
        self.DOMINATION = [1,2,3,4,5,6,10,31,32,33,34,35,36,48,65,66,68,71,74]
        self.DOMINATION.extend([75])
        self.INDEMPOTENCE = [45,46,47,49,60,70]
        self.DOUBLE_NEGATION = [58]
        self.COMMUTATIVITY = [15,20]
        self.ASSOCIATIVITY = [16,17,41,42,21,22,43,44]
        self.DISTRIBUTIVITY = [54,56,55,57]
        self.NEGATION = [7,8,9,37,38,39,50,67,76]
        self.DEMORGAN = [18,24,28,29]
        self.ABSORPTION = [64]
        self.ALL = [i for i in range(1, cur_max_op_id+1)]

        self.op_optns_diict = {'IDENTITY':self.IDENTITY,
        'BOOLEAN_EQUIVALENCE':self.BOOLEAN_EQUIVALENCE,
        'IMP_TO_DISJ':self.IMP_TO_DISJ,
        'DOMINATION':self.DOMINATION,
        'INDEMPOTENCE':self.INDEMPOTENCE,
        'DOUBLE_NEGATION':self.DOUBLE_NEGATION,
        'COMMUTATIVITY':self.COMMUTATIVITY,
        'ASSOCIATIVITY':self.ASSOCIATIVITY,
        'DISTRIBUTIVITY':self.DISTRIBUTIVITY,
        'NEGATION':self.NEGATION,
        'DEMORGAN':self.DEMORGAN,
        'ABSORPTION':self.ABSORPTION,
        'ALL':self.ALL}


        self.op_seq = op_seq


        self.expansive_ops = set([1,2,3,4,5,6,7,8,9,10,11,12,13,14,19,69])
        self.expansive_ops = self.expansive_ops.union(set([31,32,33,34,35,36]))
        self.expansive_ops = self.expansive_ops.union(set([27,25,23,30,47,51]))
        self.expansive_ops = self.expansive_ops.union(set([37,38,39,40,45,46]))
        self.expansive_ops = self.expansive_ops.union(set([62,65,66,67,70,74]))
        self.expansive_ops = self.expansive_ops.union(set([75,76]))
        self.reductive_ops = set([26,48,49,50,52,53,58,59,62,60,64,71])


        self.op_pairs_dict = {}
        if op_pairs:
            op_tups_lst=[(15,15),(20,20),(16,17),(21,22),(41,42),(43,44),
                        (23,26),(18,29),(24,28),(54,57),(55,56),(45,49),
                        (46,49),(47,49),(50,7),(50,8),(50,9),(40,53),(51,52),
                        (51,40),(60,10),(61,63)]
            for tup in op_tups_lst:
                self.op_pairs_dict[tup[0]] = tup[1]
                self.op_pairs_dict[tup[1]] = tup[0]

    def __str__(self):
        return self.parse_tree()

    def set_root(self, node):
        node.parent = self
        self.root = node

    def construct_tree(self, logexpr):
        lexer = LogicLexer()
        parser = LogicParser()
        ast = parser.parse(lexer.tokenize(logexpr))

        def construct_helper(ast_node):
            if isinstance(ast_node, Lit):
                if ast_node.value == 'T':
                    tnode = TrueNode(None)
                    return tnode
                elif ast_node.value == 'F':
                    fnode = FalseNode(None)
                    return fnode

            elif isinstance(ast_node, Var):
                if ast_node.value == 'p':
                    pnode = PNode(None)
                    return pnode
                elif ast_node.value == 'q':
                    qnode = QNode(None)
                    return qnode
                elif ast_node.value == 'r':
                    rnode = RNode(None)
                    return rnode
                elif ast_node.value == 's':
                    snode = SNode(None)
                    return snode

            elif isinstance(ast_node, NotOp):
                notnode = NotNode(None)
                argnode = construct_helper(ast_node.operand)
                notnode.set_arg(argnode)
                return notnode

            elif isinstance(ast_node, DblimpOp):
                dblimpnode = DblimplicationNode(None)
                leftargnode = construct_helper(ast_node.left)
                rightargnode = construct_helper(ast_node.right)
                dblimpnode.set_lr(leftargnode, rightargnode)
                return dblimpnode

            elif isinstance(ast_node, ImpOp):
                impnode = ImplicationNode(None)
                leftargnode = construct_helper(ast_node.left)
                rightargnode = construct_helper(ast_node.right)
                impnode.set_lr(leftargnode, rightargnode)
                return impnode

            elif isinstance(ast_node, AndOp):
                andnode = AndNode(None)
                operands = [construct_helper(oprnd) for oprnd in ast_node.operands]
                andnode.set_operands(operands)
                return andnode

            elif isinstance(ast_node, OrOp):
                ornode = OrNode(None)
                operands = [construct_helper(oprnd) for oprnd in ast_node.operands]
                ornode.set_operands(operands)
                return ornode

            elif isinstance(ast_node, Pnth):
                return construct_helper(ast_node.exp)

            else:
                print("error constructing tree")
                print("ast_node type:", type(ast_node))
                print("ast_node:", ast_node)
                sys.exit()

        self.set_root(construct_helper(ast))

    def set_computed_ops(self, ops):
        self.computed_ops = ops

    def copy(self):
        new_tree = LogicTree()
        new_tree.set_root(self.root.copy())
        new_tree.set_computed_ops(self.computed_ops)
        new_tree.op_seq = self.op_seq
        new_tree.op_pairs_dict=self.op_pairs_dict
        return new_tree

    def deep_ops(self,expand):
        # deleted_objs = gc.collect()
        # print("Doing deep_ops, deleted",deleted_objs,"objects")

        original_tree = self.copy()
        self.computed_ops += 1

        new_trees = []
        prevent_blowup_ops = set([1,2,3,4,5,6,10,11,12,13,14,19,69])
        prevent_blowup_ops = prevent_blowup_ops.union(set([25,27,30,45,46,47]))
        prevent_blowup_ops = prevent_blowup_ops.union(set([31,32,33,34,35,36]))
        prevent_blowup_ops = prevent_blowup_ops.union(set([65,66,67,74,75]))
        # prevent_blowup_ops = prevent_blowup_ops.union(set([18,24]))
        self.blowup_control = False #this is a temporary hack, remove this later

        allowed_ops = set()

        if expand==True:
            allowed_ops = self.all_ops-self.reductive_ops
        elif expand==False:
            allowed_ops = self.all_ops-self.expansive_ops
        else:
            allowed_ops = self.all_ops


        def deep_ops_helper(node, full_tree):
            nonlocal allowed_ops
            if self.op_seq:
                if(len(self.op_seq) < self.computed_ops):
                    return
                allowed_ops = allowed_ops.intersection(set(self.op_optns_diict[\
                                            self.op_seq[self.computed_ops-1]]))

            if self.blowup_control and self.computed_ops >= 2:
                passed_ops = allowed_ops - prevent_blowup_ops
            else:
                passed_ops = allowed_ops

            if isinstance(node, PNode) or \
                isinstance(node, QNode) or \
                isinstance(node, RNode) or \
                isinstance(node, SNode):
                pass
                # MAY NEED TO HAVE A BASE CASE HERE...

            # IF THE NODE IS THE LOGICTREE
            elif isinstance(node, LogicTree):
                # print("checking logic tree")
                original_expr = full_tree.parse_tree()
                original_root = node.root.copy()
                new_nodes = node.root.do_ops(passed_ops, self.op_pairs_dict)
                for new_node in new_nodes:
                    node.set_root(new_node[0])
                    if full_tree.parse_tree() != original_expr:
                        new_trees.append((full_tree.copy(), new_node[1]))

                node.set_root(original_root)
                deep_ops_helper(node.root, full_tree)

            elif isinstance(node, UnaryNode):
                # print("checking unary node")
                original_expr = full_tree.parse_tree()
                original_arg = node.arg.copy()
                new_nodes = node.arg.do_ops(passed_ops, self.op_pairs_dict)
                for new_node in new_nodes:
                    node.set_arg(new_node[0])
                    if full_tree.parse_tree() != original_expr:
                        new_trees.append((full_tree.copy(), new_node[1]))

                node.set_arg(original_arg)
                deep_ops_helper(node.arg, full_tree)

            elif isinstance(node, BinaryNode):
                # print("checking binary node")
                original_expr = full_tree.parse_tree()
                original_left = node.left.copy()
                new_left_nodes = node.left.do_ops(passed_ops, self.op_pairs_dict)
                for new_node in new_left_nodes:
                    node.set_lr(new_node[0], node.right)
                    if full_tree.parse_tree() != original_expr:
                        new_trees.append((full_tree.copy(), new_node[1]))
                node.set_lr(original_left, node.right)

                original_right = node.right.copy()
                new_right_nodes = node.right.do_ops(passed_ops, self.op_pairs_dict)
                for new_node in new_right_nodes:
                    node.set_lr(node.left, new_node[0])
                    if full_tree.parse_tree() != original_expr:
                        new_trees.append((full_tree.copy(), new_node[1]))
                node.set_lr(node.left, original_right)

                deep_ops_helper(node.left, full_tree)
                deep_ops_helper(node.right, full_tree)

            elif isinstance(node, N_aryNode):
                # print("checking n_ary node")
                original_expr = full_tree.parse_tree()
                original_operands = node.operands

                for i in range(len(original_operands)):
                    original_operand = original_operands[i]
                    # replacement_operands = original_operand.do_ops()
                    # if self.computed_ops >= 3:
                    #     replacement_operands = [node for node in replacement_operands if node[1] not in prevent_blowup_ops]

                    replacement_operands = original_operand.do_ops(passed_ops, self.op_pairs_dict)
                    for rep in replacement_operands:
                        new_operands = [op.copy() for op in original_operands]
                        new_operands[i] = rep[0]
                        node.set_operands(new_operands)
                        if full_tree.parse_tree() != original_expr:
                            new_trees.append((full_tree.copy(), rep[1]))

                node.set_operands(original_operands)
                for operand in original_operands:
                    deep_ops_helper(operand, full_tree)

        deep_ops_helper(self, self)
        return new_trees

    def parse_tree(self):
        if isinstance(self.root, BinaryNode) or isinstance(self.root, N_aryNode):
            tmp_str = self.root.parse()
            if tmp_str[0] == '(' and tmp_str[-1] == ')':
                return tmp_str[1:-1]
        else:
            return self.root.parse()

    def deep_parse_tree(self):
        if isinstance(self.root, BinaryNode) or isinstance(self.root, N_aryNode):
            tmp_strs = self.root.deep_parse()
            ret_strs = []
            for tmp_str in tmp_strs:
                if tmp_str[0] == '(' and tmp_str[-1] == ')':
                    ret_strs.append(tmp_str[1:-1])
                    ret_strs.append(tmp_str)

            return ret_strs
        else:
            return self.root.deep_parse()

    def make_gexf(self):

        def make_node_list(node):
            if isinstance(node, LeafNode):
                return [node]
            elif isinstance(node, UnaryNode):
                return [node]+make_node_list(node.arg)
            elif isinstance(node, BinaryNode):
                return [node]+make_node_list(node.left)+\
                                make_node_list(node.right)
            elif isinstance(node, N_aryNode):
                ret_lst = [node]
                for op in node.operands:
                    ret_lst += make_node_list(op)
                return ret_lst
            else:
                sys.exit()

        node_lst = make_node_list(self.root)
        graph_no = 1
        for node in node_lst:
            node.graph_no = graph_no
            graph_no += 1


        G = nx.DiGraph()
        def make_helper(lt_node):
            if isinstance(lt_node, LeafNode):
                G.add_node(lt_node.graph_no, token=lt_node.token)
            elif isinstance(lt_node, UnaryNode):
                G.add_node(lt_node.graph_no, token=lt_node.token)
                make_helper(lt_node.arg)
                G.add_edge(lt_node.graph_no, lt_node.arg.graph_no)
            elif isinstance(lt_node, BinaryNode):
                G.add_node(lt_node.graph_no, token=lt_node.token)
                make_helper(lt_node.left)
                make_helper(lt_node.right)
                G.add_edge(lt_node.graph_no, lt_node.left.graph_no)
                G.add_edge(lt_node.graph_no, lt_node.right.graph_no)
            elif isinstance(lt_node, N_aryNode):
                G.add_node(lt_node.graph_no, token=lt_node.token)
                for op in lt_node.operands:
                    make_helper(op)
                    G.add_edge(lt_node.graph_no, op.graph_no)
            else:
                sys.exit()

        make_helper(self.root)
        return G

    def make_sympy(self):

        p,q,r,s = symbols('p,q,r,s')

        def make_helper(node):

            if isinstance(node, PNode):
                return p
            elif isinstance(node, QNode):
                return q
            elif isinstance(node, RNode):
                return r
            elif isinstance(node, SNode):
                return s
            elif isinstance(node, TrueNode):
                return sympy.true
            elif isinstance(node, FalseNode):
                return sympy.false
            elif isinstance(node, NotNode):
                return Not(make_helper(node.arg))
            elif isinstance(node, OrNode):
                oplst = []
                for op in node.operands:
                    oplst.append(make_helper(op))
                return Or(*oplst)
            elif isinstance(node, AndNode):
                oplst = []
                for op in node.operands:
                    oplst.append(make_helper(op))
                return And(*oplst)
            elif isinstance(node, ImplicationNode):
                return Implies(node.left, node.right)
            elif isinstance(node, DblimplicationNode):
                return And(node.left>>node.right, node.left<<node)

        return make_helper(self.root)




class LogicTreeTrainer():

    def __init__(self, first_tree=None, expand=True, op_seq=None, op_pairs=True):

        if first_tree != None:
            first_tree = first_tree.replace('->','→')
            self.starting_expr = first_tree
            self.op_seq = op_seq
            # tree_postfix = self.inToPostFix(first_tree)
            # tree = LogicTree(tree_postfix)
            tree = LogicTree(self.starting_expr, op_seq=self.op_seq, op_pairs=op_pairs)
            self.trees = {1 : (tree, [(tree.copy(), 0)])}
            print(self.trees[1][0].parse_tree())

        else:
            self.starting_expr = 'T'
            tree = LogicTree(self.starting_expr, op_seq=self.op_seq, op_pairs=op_pairs)
            self.trees = {1 : (tree, [(tree.copy(), 0)])}

        self.expand = expand

        self.ops = 0



    def inToPostFix(self, s):

        def reject(what): # Produce a readable error
            raise SyntaxError("Expected {}, but got {} at index {}".format(
                what or "EOF",
                "'{}'".format(tokens[-1]) if tokens else "EOF",
                len(s) - len(tokens)
            ))

        get = lambda: tokens.pop() if tokens else ""
        put = lambda token: output.append(token)
        match = lambda what: tokens[-1] in what if tokens else what == ""
        expect = lambda what: get() if match(what) else reject(what)

        def suffix():
            token = get()
            term()
            put(token)

        def parens():
            expect("(")
            expression(")")

        def term():
            if match(identifier): put(get())
            elif match(unary): suffix()
            elif match("("): parens()
            else: expect("an identifier, a unary operator or an opening parenthesis");

        def expression(terminator):
            term()
            if match(binary): suffix()
            expect(terminator)

        # Define the token groups
        identifier = "abcdefghijklmnopqrstuwxyz"
        identifier += identifier.upper()
        unary = "~";
        binary = "^∧v∨→";
        # n_ary = "^∧v∨";
        tokens = list(reversed(s)) # More efficient to pop from the end
        output = [] # Will be populated during the parsing
        expression("") # Parse!
        return "".join(output)


    def increment_ops(self, ops=1):

        for i in range(ops):
            new_tree_dict = copy.deepcopy(self.trees)
            max_op_num = len(new_tree_dict[len(new_tree_dict)][1])
            for (id, treetup) in self.trees.items():
                if len(treetup[1]) == max_op_num:
                    tree = treetup[0]
                    new_trees = tree.deep_ops(self.expand)
                    for new_tree in new_trees:
                        new_sequence = treetup[1].copy()
                        new_sequence.append((new_tree[0], new_tree[1]))
                        new_tree_dict[max(new_tree_dict.keys()) + 1] = (new_tree[0], new_sequence)

            # print(len(new_tree_dict))
            # REMOVING DUPLICATES (WARNING, ALSO REMOVES CORRECTLY FORMED DUPLICATES)
            # exprs = set()
            # idx = 1
            # unduplicated_new_trees = dict()
            # i = 1
            # for id, treetup in new_tree_dict.items():
            #     expr = treetup[0].parse_tree()
            #     if expr not in exprs:
            #         exprs.add(expr)
            #         unduplicated_new_trees[idx] = treetup
            #         idx += 1
            # new_tree_dict = unduplicated_new_trees





            self.ops += 1
            self.trees = new_tree_dict
            print(len(self.trees))

            # deleted = gc.collect()
            # print("Doing increment_ops, deleted",deleted,"objects")


    def duplicates_info(self):

        dataset = self.get_tree_sequences()
        trees = []
        for i in range(len(dataset)):
            treetup = dataset[i]
            trees.append((treetup[0].parse_tree(), len(treetup[1]) - 1, treetup[1]))

        max_op_num = self.ops
        op_depths_dict = {i:[] for i in range(max_op_num+1)}

        for treetup in trees:
            op_depths_dict[treetup[1]].append(treetup)

        all_duplicates = []
        for depth in range(len(op_depths_dict)):

            dup_dict = dict()
            checktrees = op_depths_dict[depth]
            passed_strs = []
            for treetup in checktrees:
                seq = []
                for tree in treetup[2]:
                    seq.append(tree[0].parse_tree())

                if treetup[0] in passed_strs:
                    if treetup[0] in dup_dict.keys():
                        dup_dict[treetup[0]].append(seq)
                    else:
                        dup_dict[treetup[0]] = [seq]
                else:
                    dup_dict[treetup[0]] = [seq]
                    passed_strs.append(treetup[0])

            dup_dict2 = dict()
            for expr, occurences in dup_dict.items():
                if len(occurences) > 1:
                    dup_dict2[expr] = occurences

            all_duplicates.append(dup_dict2)

            print(depth, len(dup_dict2)/len(checktrees))


        return all_duplicates


    def cross_depth_dups_info(self, prev, cur):
        dataset = self.get_tree_sequences()
        trees = []
        for i in range(len(dataset)):
            treetup = dataset[i]
            trees.append((treetup[0].parse_tree(), len(treetup[1]) - 1, treetup[1]))

        max_op_num = self.ops
        op_depths_dict = {i:[] for i in range(max_op_num+1)}

        for treetup in trees:
            op_depths_dict[treetup[1]].append(treetup)

        depth_strs = [[] for i in range(len(op_depths_dict))]

        for i in range(len(depth_strs)):
            full_list = op_depths_dict[i]
            depth_strs[i].extend([treetup[0] for treetup in full_list])

        cur_strs = set(depth_strs[cur])
        prev_strs = set(depth_strs[prev])

        all_results_tup = []
        cur_full_lists = op_depths_dict[cur]
        prev_full_lists = op_depths_dict[prev]
        for expr in list(cur_strs.intersection(prev_strs)):
            results_tup = ([],[])
            for treetup in prev_full_lists:
                if treetup[0] == expr:
                    results_tup[0].append(treetup)
            for treetup in cur_full_lists:
                if treetup[0] == expr:
                    results_tup[1].append(treetup)

            all_results_tup.append(results_tup)

        return all_results_tup

        # return list(cur_strs.intersection(prev_strs))


    def get_trees(self):
        return [tup[0] for tup in list(self.trees.values())]
    def get_sequences(self):
        return [tup[1] for tup in list(self.trees.values())]
    def get_tree_sequences(self):
        return [tup for tup in list(self.trees.values())]

    def write_gexfs(self, outdir=None):
        if not outdir:
            outdir = '../data/gexfs/T/'
        labs_file = open(outdir+'labels.txt', 'w', encoding='utf8')
        out_file_no = 1
        for (tree, seq) in self.trees.values():
            G = tree.make_gexf()
            out_file_name = outdir+str(out_file_no)+'.gexf'
            label = str(out_file_no)+' : '+tree.parse_tree()+'\n'
            labs_file.write(label)
            out_file_no += 1
            nx.write_gexf(G,out_file_name)

        labs_file.close()
        pkl.dump(self, open(outdir+'trainer.pkl', 'wb'))

    def swap_or(self, expr):
        symbols = list(expr)
        or_inds = []
        for i,c in enumerate(symbols):
            if c == 'v' or c == '∨':
                or_inds.append(i)
        random.shuffle(or_inds)
        symbols[or_inds[0]] = '∧'
        return ''.join(symbols)

    def swap_and(self, expr):
        symbols = list(expr)
        and_inds = []
        for i,c in enumerate(symbols):
            if c == '^' or c == '∧':
                and_inds.append(i)
        random.shuffle(and_inds)
        symbols[and_inds[0]] = '∨'
        return ''.join(symbols)

    def swap_p_q(self, expr):
        symbols = list(expr)
        p_inds = []
        for i,c in enumerate(symbols):
            if c == 'p':
                p_inds.append(i)
        random.shuffle(p_inds)
        symbols[p_inds[0]] = 'q'
        return ''.join(symbols)

    def swap_q_p(self, expr):
        symbols = list(expr)
        q_inds = []
        for i,c in enumerate(symbols):
            if c == 'q':
                q_inds.append(i)
        random.shuffle(q_inds)
        symbols[q_inds[0]] = 'p'
        return ''.join(symbols)

    def str_mistakes(self, pfs, num_mistakes=1, final_exp=None, ok_mistks=None):
        if not pfs:
            return []

        mistk_types = {1:"swap v for ^",
                        2:"swap ^ for v",
                        3:"swap p for q",
                        4:"swap q for p"}

        # pfs = self.get_sequences()
        if final_exp:
            t_pfs = []
            for pf in pfs:
                if pf[-1][0].parse_tree() == final_exp:
                    t_pfs.append(pf)
            if len(t_pfs) == 0:
                print("no good proofs")
                sys.exit()
            pfs = t_pfs
        pf_strs = []
        for sequ in pfs:
            new_seq = []
            for tup in sequ:
                new_seq.append((tup[0].parse_tree(), tup[1]))
            pf_strs.append(new_seq)


        mistake_proofs = []
        while len(mistake_proofs) < num_mistakes:
            pf_ind = random.randint(0,len(pf_strs)-1)
            # mistk_pf = pf_strs[pf_ind]

            mistk_pf = []
            for t in pf_strs[pf_ind]:
                mistk_pf.append((copy.copy(t[0]), copy.copy(t[1])))

            if len(mistk_pf) == 1:
                continue

            mistk_step = random.randint(1,len(mistk_pf)-1)
            if ok_mistks:
                random.shuffle(ok_mistks)
                mistk_type = ok_mistks[0]
            else:
                mistk_type = random.randint(1, len(mistk_types))

            if mistk_type == 1:
                if '∨' in mistk_pf[mistk_step][0]:
                    new_expr = self.swap_or(mistk_pf[mistk_step][0])
                    mistk_pf[mistk_step] = (new_expr, (mistk_pf[mistk_step][1],
                                            mistk_types[mistk_type]))
                    mistake_proofs.append(mistk_pf)
                else:
                    continue
            elif mistk_type == 2:
                if '∧' in mistk_pf[mistk_step][0]:
                    new_expr = self.swap_and(mistk_pf[mistk_step][0])
                    mistk_pf[mistk_step] = (new_expr, (mistk_pf[mistk_step][1],
                                            mistk_types[mistk_type]))
                    mistake_proofs.append(mistk_pf)
                else:
                    continue
            elif mistk_type == 3:
                temp_true = TrueNode(None)
                if 'p' in mistk_pf[mistk_step][0] and \
                            mistk_pf[mistk_step][1] not in temp_true.nodeOps:
                    new_expr = self.swap_p_q(mistk_pf[mistk_step][0])
                    mistk_pf[mistk_step] = (new_expr, (mistk_pf[mistk_step][1],
                                            mistk_types[mistk_type]))
                    mistake_proofs.append(mistk_pf)
                else:
                    continue
            elif mistk_type == 4:
                temp_true = TrueNode(None)
                if 'q' in mistk_pf[mistk_step][0] and \
                            mistk_pf[mistk_step][1] not in temp_true.nodeOps:
                    new_expr = self.swap_q_p(mistk_pf[mistk_step][0])
                    # mistk_pf[mistk_step] = (new_expr, (mistk_type, mistk_types[mistk_type]))
                    mistk_pf[mistk_step] = (new_expr, (mistk_pf[mistk_step][1],
                                            mistk_types[mistk_type]))
                    mistake_proofs.append(mistk_pf)
                else:
                    continue


        return mistake_proofs

    def remove_not(self, orig_tree):
        cp_tree = orig_tree.copy()
        prnt = cp_tree
        child = cp_tree.root

        def find_nots(prnt, child):
            if isinstance(child, LeafNode):
                return []

            elif isinstance(child, NotNode):
                return [(prnt,child)] + find_nots(child, child.arg)
            elif isinstance(child, BinaryNode):
                return find_nots(child, child.left) + find_nots(child, child.right)
            elif isinstance(child, N_aryNode):
                nots = []
                for op in child.operands:
                    nots.extend(find_nots(child, op))
                return nots
            else:
                print("Couldn't find Not")
                sys.exit()


        all_nots = find_nots(prnt, child)
        if all_nots:
            random.shuffle(all_nots)
            tup = all_nots[0]
            new_child = tup[1].arg.copy()

            if isinstance(tup[0], LogicTree):
                tup[0].set_root(new_child)
            elif isinstance(tup[0], BinaryNode):
                if tup[0].left is tup[1]:
                    tup[0].set_lr(new_child, tup[0].right)
                elif tup[0].right is tup[1]:
                    tup[0].set_lr(tup[0].left, new_child)
                else:
                    print("huston, problem here")
                    sys.exit()
            elif isinstance(tup[0], N_aryNode):
                for i in range(len(tup[0].operands)):
                    if tup[0].operands[i] is tup[1]:
                        new_ops = [op.copy() for op in tup[0].operands]
                        new_ops[i] = new_child
                        tup[0].set_operands(new_ops)
                        break
                else:
                    print(tup[0].parse())
                    print("huston, problem now")
                    sys.exit()
            elif isinstance(tup[0], NotNode):
                tup[0].arg = new_child

            return cp_tree.copy()

    def demorgan_mistake1(self, orig_tree):
        # Normal: ~p^~q = ~(pvq)
        # Mistake: ~p^~q = ~(p^q)
        cp_tree = orig_tree.copy()
        prnt = cp_tree
        child = cp_tree.root

        def find_demorg(prnt, child):
            if isinstance(child, LeafNode):
                # print("leaf", child.parse())
                return []
            elif isinstance(child, BinaryNode):
                # print("bin", child.parse())
                return find_demorg(child, child.left)+find_demorg(child, child.right)
            elif isinstance(child, N_aryNode):
                # print("n", child.parse())
                dems = []
                for op in child.operands:
                    dems += find_demorg(child, op)
                return dems
            elif isinstance(child, NotNode):
                # print("not", child.parse())
                if child.most_recent_op == 18 and \
                    isinstance(child.arg,OrNode) and\
                    len(child.arg.operands) == 2:
                    return [(prnt, child)] + find_demorg(child, child.arg)
                return []
            else:
                return []

        demorgs = find_demorg(prnt, child)
        if demorgs:
            random.shuffle(demorgs)
            tup = demorgs[0]
            assert(isinstance(tup[1].arg,OrNode) and len(tup[1].arg.operands) == 2)

            new_child = tup[1].copy()
            andnode = AndNode(new_child)
            new_ops = [o.copy() for o in tup[1].arg.operands]
            andnode.set_operands(new_ops)
            new_child.set_arg(andnode)

            if isinstance(tup[0], LogicTree):
                if tup[0].root is tup[1]:
                    tup[0].set_root(new_child)
            elif isinstance(tup[0], BinaryNode):
                if tup[0].left is tup[1]:
                    tup[0].set_lr(new_child, tup[0].right)
                elif tup[0].right is tup[1]:
                    tup[0].set_lr(tup[0].left, new_child)
                else:
                    print("huston, problem here")
                    sys.exit()
            elif isinstance(tup[0], N_aryNode):
                for i in range(len(tup[0].operands)):
                    if tup[0].operands[i] is tup[1]:
                        new_ops = [op.copy() for op in tup[0].operands]
                        new_ops[i] = new_child
                        tup[0].set_operands(new_ops)
                        break
                else:
                    print(tup[0].parse())
                    print("huston, problem now")
                    sys.exit()
            elif isinstance(tup[0], NotNode):
                if tup[0].arg is tup[1]:
                    tup[0].arg = new_child

            else:
                print("huh?")
                sys.exit()

            return cp_tree.copy()

        else:
            return None

    def demorgan_mistake2(self, orig_tree):
        # Normal: ~(pvq) = ~p^~q
        # Mistake: ~(pvq) = ~pv~q
        # Note: since the normal operation yeilds an and node with two operands
        #       we can just return an or node and not worry about n_ary and
        cp_tree = orig_tree.copy()
        prnt = cp_tree
        child = cp_tree.root

        def find_demorg(prnt, child):
            if isinstance(child, LeafNode):
                # print("leaf", child.parse())
                return []
            elif isinstance(child, BinaryNode):
                # print("bin", child.parse())
                return find_demorg(child, child.left)+find_demorg(child, child.right)
            elif isinstance(child, N_aryNode):
                # print("n", child.parse())
                dems = []
                if child.most_recent_op == 29:
                    dems.append((prnt, child))
                for op in child.operands:
                    dems += find_demorg(child, op)
                return dems
            elif isinstance(child, NotNode):
                # print("not", child.parse())
                return find_demorg(child, child.arg)
            else:
                return []

        demorgs = find_demorg(prnt, child)
        if demorgs:
            random.shuffle(demorgs)
            tup = demorgs[0]
            # assert(isinstance(tup[1],AndNode) and len(tup[1].operands) == 2)

            new_child = OrNode(tup[0])
            new_ops = [o.copy() for o in tup[1].operands]
            new_child.set_operands(new_ops)

            if isinstance(tup[0], LogicTree):
                if tup[0].root is tup[1]:
                    tup[0].set_root(new_child)
                else:
                    print("huston, big problem")
            elif isinstance(tup[0], BinaryNode):
                if tup[0].left is tup[1]:
                    tup[0].set_lr(new_child, tup[0].right)
                elif tup[0].right is tup[1]:
                    tup[0].set_lr(tup[0].left, new_child)
                else:
                    print("huston, problem here")
                    sys.exit()
            elif isinstance(tup[0], N_aryNode):
                for i in range(len(tup[0].operands)):
                    if tup[0].operands[i] is tup[1]:
                        new_ops = [op.copy() for op in tup[0].operands]
                        new_ops[i] = new_child
                        tup[0].set_operands(new_ops)
                        break
                else:
                    print(tup[0].parse())
                    print("huston, problem now")
                    sys.exit()
            elif isinstance(tup[0], NotNode):
                if tup[0].arg is tup[1]:
                    tup[0].arg = new_child

            else:
                print("huh?")
                sys.exit()

            return cp_tree.copy()

        else:
            return None

    def demorgan_mistake3(self, orig_tree):
        # Normal: ~(pvq) = ~p^~q
        # Mistake: ~(pvq) = p^q
        cp_tree = orig_tree.copy()
        prnt = cp_tree
        child = cp_tree.root

        def find_demorg(prnt, child):
            if isinstance(child, LeafNode):
                # print("leaf", child.parse())
                return []
            elif isinstance(child, BinaryNode):
                # print("bin", child.parse())
                return find_demorg(child, child.left)+find_demorg(child, child.right)
            elif isinstance(child, N_aryNode):
                # print("n", child.parse())
                dems = []
                if child.most_recent_op == 29:
                    dems.append((prnt, child))
                for op in child.operands:
                    dems += find_demorg(child, op)
                return dems
            elif isinstance(child, NotNode):
                # print("not", child.parse())
                return find_demorg(child, child.arg)
            else:
                return []

        demorgs = find_demorg(prnt, child)
        if demorgs:
            random.shuffle(demorgs)
            tup = demorgs[0]
            assert(isinstance(tup[1],AndNode) and len(tup[1].operands) == 2)

            new_child = tup[1].copy()
            new_ops = [new_child.operands[0].arg, new_child.operands[1].arg]
            new_child.set_operands(new_ops)

            if isinstance(tup[0], LogicTree):
                if tup[0].root is tup[1]:
                    tup[0].set_root(new_child)
                else:
                    print("huston, big problem")
            elif isinstance(tup[0], BinaryNode):
                if tup[0].left is tup[1]:
                    tup[0].set_lr(new_child, tup[0].right)
                elif tup[0].right is tup[1]:
                    tup[0].set_lr(tup[0].left, new_child)
                else:
                    print("huston, problem here")
                    sys.exit()
            elif isinstance(tup[0], N_aryNode):
                for i in range(len(tup[0].operands)):
                    if tup[0].operands[i] is tup[1]:
                        new_ops = [op.copy() for op in tup[0].operands]
                        new_ops[i] = new_child
                        tup[0].set_operands(new_ops)
                        break
                else:
                    print(tup[0].parse())
                    print("huston, problem now")
                    sys.exit()
            elif isinstance(tup[0], NotNode):
                if tup[0].arg is tup[1]:
                    tup[0].arg = new_child

            else:
                print("huh?")
                sys.exit()

            return cp_tree.copy()

        else:
            return None

    def demorgan_mistake4(self, orig_tree):
        # normally: ~(p^q) = ~pv~q
        # mistake: ~(p^q) = ~p^~q
        cp_tree = orig_tree.copy()
        prnt = cp_tree
        child = cp_tree.root

        def find_demorg(prnt, child):
            if isinstance(child, LeafNode):
                # print("leaf", child.parse())
                return []
            elif isinstance(child, BinaryNode):
                # print("bin", child.parse())
                return find_demorg(child, child.left)+find_demorg(child, child.right)
            elif isinstance(child, N_aryNode):
                # print("n", child.parse())
                dems = []
                if child.most_recent_op == 28:
                    dems.append((prnt, child))
                for op in child.operands:
                    dems += find_demorg(child, op)
                return dems
            elif isinstance(child, NotNode):
                # print("not", child.parse())
                return find_demorg(child, child.arg)
            else:
                return []

        demorgs = find_demorg(prnt, child)
        if demorgs:
            random.shuffle(demorgs)
            tup = demorgs[0]
            assert(isinstance(tup[1],OrNode) and len(tup[1].operands) == 2)

            new_child = AndNode(tup[0])
            new_ops = [o.copy() for o in tup[1].operands]
            new_child.set_operands(new_ops)

            if isinstance(tup[0], LogicTree):
                if tup[0].root is tup[1]:
                    tup[0].set_root(new_child)
                else:
                    print("huston, big problem")
            elif isinstance(tup[0], BinaryNode):
                if tup[0].left is tup[1]:
                    tup[0].set_lr(new_child, tup[0].right)
                elif tup[0].right is tup[1]:
                    tup[0].set_lr(tup[0].left, new_child)
                else:
                    print("huston, problem here")
                    sys.exit()
            elif isinstance(tup[0], N_aryNode):
                for i in range(len(tup[0].operands)):
                    if tup[0].operands[i] is tup[1]:
                        new_ops = [op.copy() for op in tup[0].operands]
                        new_ops[i] = new_child
                        tup[0].set_operands(new_ops)
                        break
                else:
                    print(tup[0].parse())
                    print("huston, problem now")
                    sys.exit()
            elif isinstance(tup[0], NotNode):
                if tup[0].arg is tup[1]:
                    tup[0].arg = new_child

            else:
                print("huh?")
                sys.exit()

            return cp_tree.copy()

        else:
            return None

    def demorgan_mistake5(self, orig_tree):
        # Normal: ~pv~q = ~(p^q)
        # Mistake: ~pv~q = ~(pvq)
        cp_tree = orig_tree.copy()
        prnt = cp_tree
        child = cp_tree.root

        def find_demorg(prnt, child):
            if isinstance(child, LeafNode):
                return []
            elif isinstance(child, BinaryNode):
                return find_demorg(child, child.left)+find_demorg(child, child.right)
            elif isinstance(child, N_aryNode):
                dems = []
                for op in child.operands:
                    dems += find_demorg(child, op)
                return dems
            elif isinstance(child, NotNode):
                if child.most_recent_op == 24 and \
                    isinstance(child.arg,AndNode) and \
                    len(child.arg.operands) == 2:
                    return [(prnt, child)] + find_demorg(child, child.arg)
                return []
            else:
                return []

        demorgs = find_demorg(prnt, child)
        if demorgs:
            random.shuffle(demorgs)
            tup = demorgs[0]
            assert(isinstance(tup[1].arg,AndNode))
            assert(len(tup[1].arg.operands) == 2)

            new_child = tup[1].copy()
            ornode = OrNode(new_child)
            new_ops = [o.copy() for o in tup[1].arg.operands]
            ornode.set_operands(new_ops)
            new_child.set_arg(ornode)

            if isinstance(tup[0], LogicTree):
                if tup[0].root is tup[1]:
                    tup[0].set_root(new_child)
            elif isinstance(tup[0], BinaryNode):
                if tup[0].left is tup[1]:
                    tup[0].set_lr(new_child, tup[0].right)
                elif tup[0].right is tup[1]:
                    tup[0].set_lr(tup[0].left, new_child)
                else:
                    print("huston, problem here")
                    sys.exit()
            elif isinstance(tup[0], N_aryNode):
                for i in range(len(tup[0].operands)):
                    if tup[0].operands[i] is tup[1]:
                        new_ops = [op.copy() for op in tup[0].operands]
                        new_ops[i] = new_child
                        tup[0].set_operands(new_ops)
                        break
                else:
                    print(tup[0].parse())
                    print("huston, problem now")
                    sys.exit()
            elif isinstance(tup[0], NotNode):
                if tup[0].arg is tup[1]:
                    tup[0].arg = new_child

            else:
                print("huh?")
                sys.exit()

            return cp_tree.copy()

        else:
            return None

    def indempotence_mistake(self, orig_tree):
        # Normal: pvp = p
        # Mistake: pvp = T

        cp_tree = orig_tree.copy()
        prnt = cp_tree
        child = cp_tree.root

        def find_indemp(prnt, child):
            if isinstance(child, LeafNode):
                return []
            elif isinstance(child, BinaryNode):
                return find_indemp(child, child.left)+find_indemp(child, child.right)
            elif isinstance(child, N_aryNode):
                indmps = []
                if isinstance(child, OrNode) and child.most_recent_op == 49:
                    for i in range(len(child.operands)-1):
                        if child.operands[i] == child.operands[i+1]:
                            indmps.append((prnt,child))
                for op in child.operands:
                    indmps += find_indemp(child, op)
                return indmps
            elif isinstance(child, NotNode):
                return find_indemp(child, child.arg)
            else:
                return []

        indmps = find_indemp(prnt, child)
        if indmps:
            random.shuffle(indmps)
            tup = indmps[0]
            assert(isinstance(tup[1], OrNode))

            new_child = tup[1].copy()
            new_ops = [op.copy() for op in tup[1].operands]
            for i in range(len(new_ops)-1):
                if new_ops[i] == new_ops[i+1]:
                    tnode = TrueNode(new_child)
                    new_ops.pop(i)
                    new_ops[i] = tnode
                    break
            else:
                print("couldn't find the duplicate operand")
                sys.exit()
            new_child.set_operands(new_ops)

            if isinstance(tup[0], LogicTree):
                if tup[0].root is tup[1]:
                    tup[0].set_root(new_child)
            elif isinstance(tup[0], BinaryNode):
                if tup[0].left is tup[1]:
                    tup[0].set_lr(new_child, tup[0].right)
                elif tup[0].right is tup[1]:
                    tup[0].set_lr(tup[0].left, new_child)
                else:
                    print("huston, problem here")
                    sys.exit()
            elif isinstance(tup[0], N_aryNode):
                for i in range(len(tup[0].operands)):
                    if tup[0].operands[i] is tup[1]:
                        new_ops = [op.copy() for op in tup[0].operands]
                        new_ops[i] = new_child
                        tup[0].set_operands(new_ops)
                        break
                else:
                    print(tup[0].parse())
                    print("huston, problem now")
                    sys.exit()
            elif isinstance(tup[0], NotNode):
                if tup[0].arg is tup[1]:
                    tup[0].arg = new_child

            else:
                print("huh?")
                sys.exit()

            return cp_tree.copy()

        else:
            return None

    def node_mistakes(self, pfs, num_mistakes=1, final_exp=None, ok_mistks=None):
        if not pfs:
            return []

        mistk_types = {1:"remove ~",
                        2:"incorrect demorgans",
                        3:"incorrect indempotence"}

        if final_exp:
            t_pfs = []
            for pf in pfs:
                if pf[-1][0].parse_tree() == final_exp:
                    t_pfs.append(pf)
            if len(t_pfs) == 0:
                print("no proofs ending with",final_exp)
                sys.exit()
            pfs = t_pfs

        mistake_proofs = []
        while len(mistake_proofs) < num_mistakes:
            pf_ind = random.randint(0,len(pfs)-1)
            mistk_pf = []
            for t in pfs[pf_ind]:
                mistk_pf.append((t[0].copy(), t[1]))

            if len(mistk_pf) == 1:
                continue

            mistk_step = random.randint(1,len(mistk_pf)-1)

            if ok_mistks:
                random.shuffle(ok_mistks)
                mistk_type = ok_mistks[0]
            else:
                mistk_type = random.randint(1, len(mistk_types))


            if mistk_type == 1:
                if '~' in mistk_pf[mistk_step][0].parse_tree():
                    new_tree = self.remove_not(mistk_pf[mistk_step][0])
                    mistk_pf[mistk_step] = (new_tree, (mistk_pf[mistk_step][1],
                                                mistk_types[mistk_type]))
                    mistake_proofs.append(mistk_pf)
                else:
                    continue

            elif type(mistk_type) == tuple and mistk_type[0] == 2:
                dem_mistk_types = {1:"~p^~q becomes ~(p^q)",
                                    2:"~(pvq) becomes ~pv~q",
                                    3:"~(pvq) becomes p^q",
                                    4:"~(p^q) becomes ~p^~q",
                                    5:"~pv~q becomes ~(pvq)"}
                dem_mistk_oprs = {1:self.demorgan_mistake1,
                                        2:self.demorgan_mistake2,
                                        3:self.demorgan_mistake3,
                                        4:self.demorgan_mistake4,
                                        5:self.demorgan_mistake5}
                if mistk_type[1] == -1:
                    dem_mistk_type = random.randint(1, len(dem_mistk_types))
                else:
                    dem_mistk_type = mistk_type[1]

                new_tree = dem_mistk_oprs[dem_mistk_type](mistk_pf[mistk_step][0])
                if new_tree:
                    mistk_pf[mistk_step] = (new_tree, (mistk_pf[mistk_step][1],
                                            dem_mistk_types[dem_mistk_type]))
                    mistake_proofs.append(mistk_pf)
                else:
                    continue

            elif mistk_type == 3:
                new_tree = self.indempotence_mistake(mistk_pf[mistk_step][0])
                if new_tree:
                    mistk_pf[mistk_step] = (new_tree, (mistk_pf[mistk_step][1],
                                                mistk_types[mistk_type]))
                    mistake_proofs.append(mistk_pf)
                else:
                    continue

        return mistake_proofs

    def generate_mistakes(self, \
                            num_mistakes=5, \
                            return_strs=True, \
                            final_exp=None,\
                            verbose=False):
        all_pfs = self.get_sequences()
        all_op_seqs = []
        for pf in all_pfs:
            all_op_seqs.append([op for (_,op) in pf])

        cur_num_mistk_options = 11
        mistks_per_typ = math.ceil(int(num_mistakes/cur_num_mistk_options))

        str_mistks = []
        node_mistks = []

        node_mistk_ops = [1, (2,1), (2,2), (2,3), (2,4), (2,5), 3]

        if verbose:
            print("doing node, remove not")
        node_mistks.extend(self.node_mistakes(all_pfs,\
                                mistks_per_typ,final_exp,[1]))

        dem_mistk_1_pfs = []
        dem_mistk_2_pfs = []
        dem_mistk_3_pfs = []
        dem_mistk_4_pfs = []
        dem_mistk_5_pfs = []
        indmp_mistk_pfs = []
        for i in range(len(all_op_seqs)):
            if 18 in all_op_seqs[i]:
                dem_mistk_1_pfs.append(all_pfs[i])
            if 29 in all_op_seqs[i]:
                dem_mistk_2_pfs.append(all_pfs[i])
                dem_mistk_3_pfs.append(all_pfs[i])
            if 28 in all_op_seqs[i]:
                dem_mistk_4_pfs.append(all_pfs[i])
            if 24 in all_op_seqs[i]:
                dem_mistk_5_pfs.append(all_pfs[i])
            if 49 in all_op_seqs[i]:
                indmp_mistk_pfs.append(all_pfs[i])

        if verbose:
            print("doing node, dem1")
        node_mistks.extend(self.node_mistakes(dem_mistk_1_pfs,\
                                mistks_per_typ,final_exp,[(2,1)]))

        if verbose:
            print("doing node, dem2")
        node_mistks.extend(self.node_mistakes(dem_mistk_2_pfs,\
                                mistks_per_typ,final_exp,[(2,2)]))

        if verbose:
            print("doing node, dem3")
        node_mistks.extend(self.node_mistakes(dem_mistk_3_pfs,\
                                mistks_per_typ,final_exp,[(2,3)]))

        if verbose:
            print("doing node, dem4")
        node_mistks.extend(self.node_mistakes(dem_mistk_4_pfs,\
                                mistks_per_typ,final_exp,[(2,4)]))

        if verbose:
            print("doing node, dem5")
        node_mistks.extend(self.node_mistakes(dem_mistk_5_pfs,\
                                mistks_per_typ,final_exp,[(2,5)]))

        if verbose:
            print("doing node, indempotence")
        node_mistks.extend(self.node_mistakes(indmp_mistk_pfs,\
                                mistks_per_typ,final_exp,[3]))



        for str_mistk_type in range(1,5):
            if verbose:
                print("doing str,",str_mistk_type)
            str_mistks.extend(self.str_mistakes(all_pfs, mistks_per_typ,\
                                final_exp,[str_mistk_type]))


        if return_strs:
            ret_mistks = []
            for pf in node_mistks:
                ret_mistks.append([(tup[0].parse_tree(), tup[1]) for tup in pf])
            ret_mistks += str_mistks
            return ret_mistks
        return (str_mistks, node_mistks)




if __name__ == '__main__':

    trainer = LogicTreeTrainer('T', expand=None)
    trainer.increment_ops(1)
    #
    # save = '../data/unduped/T_unduped_mistakes.pkl'
    # pkl.dump(trainer, open(save,'wb'))

    """
    trainers = []

    starting_exprs = ['~(~p)↔p']
    for expr in starting_exprs:
        print("building trees from " + expr)
        trainer = LogicTreeTrainer(expr,expand=None)
        trainer.increment_ops(4)
        trainers.append(trainer)

    for i in range(len(trainers)):
        trainer = trainers[i]
        seed = starting_exprs[i]
        save = '../data/unduped/' + seed + '_unduped_mistakes.pkl'
        pkl.dump(trainer, open(save, 'wb'))
    """

# comment
