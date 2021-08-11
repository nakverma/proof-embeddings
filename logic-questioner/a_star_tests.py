import astar
import os
import sys
import csv
import math
import json
import time
from difflib import SequenceMatcher
import unittest
from create_expressions_mistakes import LogicTreeTrainer

def big_change_favored_weight(n1, n2):
    next = []

    seed = n1
    trainer = LogicTreeTrainer(seed, expand=None)
    trainer.increment_ops(1)

    for key in trainer.trees.keys():
        if key > 1:
            if str(trainer.trees[key][0]) == n2:
                op_code = trainer.trees[key][1][1][1]
    if op_code in [10,11,12,13,14,19,25,27,62,30,59,68,69,72,73]:
        weight = 7 #'Identity'
    elif op_code in [51,40,52,53]:
        weight = 6 # Boolean Equivalence / Literal Negation
    elif op_code in [23,26,84,85]:
        weight = 2 #'Impl to Disj'
    elif op_code in [61,63]:
        weight = 1 #'iff to Impl'
    elif op_code in [1,2,3,4,5,6,10,31,32,33,34,35,36,48,65,66,68,71,74,75]:
        weight = 7 #'Domination'
    elif op_code in [45,46,47,49,60,70]:
        weight = 7 #'Idemptotence'
    elif op_code in [58,83]:
        weight = 6 #'Double Negation'
    elif op_code in [15,20]:
        weight = 8 #'Commutativity'
    elif op_code in [16,17,41,42,21,22,43,44]:
        weight = 8 #'Associativity'
    elif op_code in [54,56,55,57,78,79,80,81]:
        weight = 4 #'Distributivity'
    elif op_code in [7,8,9,37,38,39,50,67,76,82]:
        weight = 7 #'Negation'
    elif op_code in [18,24,28,29]:
        weight = 3 # 'DeMorgan'
    elif op_code in [64,77]:
        weight = 5 #'Absorption'
    return weight

def small_change_favored_weight(n1, n2):
    return 9 - big_change_favored_weight(n1, n2)

def h1(start, next, ans):
    return abs(len(next)-len(ans))

def h2(start, next, ans):
    return abs(len(next)-len(ans)) / abs(len(start)-len(ans))

def h3(start, next, ans):
    return abs(len(start)-len(ans)) - abs(len(next)-len(ans))

def h4(start, next, ans):
    return abs(len(start)-len(ans)) - big_change_favored_weight() * abs(len(next)-len(ans))

def h5(start, next, ans):
    return big_change_favored_weight() * abs(len(next)-len(ans)) / abs(len(start)-len(ans))

def h6(start, next, ans):
    return abs(len(start)-len(ans)) - abs(len(next)-len(ans))

def g1(n1, n2):
    return big_change_favored_weight(n1, n2)

def g2(n1, n2):
    return small_change_favored_weight(n1, n2)

def g3(n1, n2, h):
    CUTOFF = 4 # Needs to be tested and changed
    if h(n1, n2) > CUTOFF:
        return big_change_favored_weight(n1, n2)
    else:
        return small_change_favored_weight(n1, n2)

def __main__():
    pass

if __name__ == '__main__':
    __main__()
