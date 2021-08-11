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
    op_code = 0
    weight = 0

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
    return abs(len(start)-len(ans)) - big_change_favored_weight(start, next) * abs(len(next)-len(ans))

def h5(start, next, ans):
    return big_change_favored_weight(start, next) * abs(len(next)-len(ans)) / abs(len(start)-len(ans))

def h6(start, next, ans):
    return abs(len(start)-len(ans)) - abs(len(next)-len(ans))

def g1(n1, n2):
    return big_change_favored_weight(n1, n2)

def g2(n1, n2):
    return small_change_favored_weight(n1, n2)

def g3(n1, n2,  ans, h):
    CUTOFF = 4 # Needs to be tested and changed
    if h(n1, n2, ans) > CUTOFF:
        return big_change_favored_weight(n1, n2)
    else:
        return small_change_favored_weight(n1, n2)


def sample_questions(question_file):
    with open(question_file) as file:
        questions = json.load(file)
    print('Start\tEnd\tCurrent\tNext\tg1\tg2\tg3\th1\th2\th3\th4\th5\th6')
    with open('output.txt','w') as file:
        max_col_length = {'start': 25, 'ans': 15, 'current': 25, 'next': 25, 'func': 15}
        for i, question in enumerate(questions):
            start = question['question'].split('that ')[1].split(' is')[0]
            ans = question['answer']
            steps = question['solution']
            g_segment_1 = [g1(steps[i], steps[i+1]) for i in range(len(steps)-1)]
            g_segment_2 = [g2(steps[i], steps[i+1]) for i in range(len(steps)-1)]
            g_total_1 = [g1(steps[i], steps[i+1]) for i in range(len(steps)-1)]
            g_total_2 = [g2(steps[i], steps[i+1]) for i in range(len(steps)-1)]
            for i in reversed(range(len(steps)-2)):
                g_total_1[i] = g_total_1[i] if i == len(steps)-1 else g_total_1[i] + g_total_1[i+1]
                g_total_2[i] = g_total_2[i] if i == len(steps)-1 else g_total_2[i] + g_total_2[i+1]
            for i in range(len(steps)-1):
                row = start + ' ' * (max_col_length['start']-len(start))
                row += ans + ' ' * (max_col_length['ans']-len(ans))
                row += steps[i] + ' ' * (max_col_length['current']-len(steps[i]))
                row += steps[i+1] + ' ' * (max_col_length['next']-len(steps[i+1]))
                row += str(g_total_1[i]) + ' ' * (max_col_length['func']-len(str(g_total_1[i])))
                row += str(g_total_2[i]) + ' ' * (max_col_length['func']-len(str(g_total_2[i])))
                row += str(h1(start, steps[i+1], ans)) + ' ' * (max_col_length['func']-len(str(h1(start, steps[i+1], ans))))
                row += str(h2(start, steps[i+1], ans)) + ' ' * (max_col_length['func']-len(str(h2(start, steps[i+1], ans))))
                row += str(h3(start, steps[i+1], ans)) + ' ' * (max_col_length['func']-len(str(h3(start, steps[i+1], ans))))
                row += str(h4(start, steps[i+1], ans)) + ' ' * (max_col_length['func']-len(str(h4(start, steps[i+1], ans))))
                row += str(h5(start, steps[i+1], ans)) + ' ' * (max_col_length['func']-len(str(h5(start, steps[i+1], ans))))
                row += str(h6(start, steps[i+1], ans)) + ' ' * (max_col_length['func']-len(str(h6(start, steps[i+1], ans))))
                file.write(row + '\n')

def __main__():
    start = 'qvp'
    ans = '~p->q'
    nxt = '~p->q'
    print("g:")
    print(g1(start,nxt))
    print(g2(start,nxt))
    print(g3(start,nxt, ans, h1))
    print(g3(start,nxt, ans,h2))
    print(g3(start,nxt, ans,h3))
    print(g3(start,nxt, ans,h4))
    print(g3(start,nxt, ans,h5))
    print(g3(start,nxt, ans,h6))
    print("h:")
    print(h1(start, nxt, ans))
    print(h2(start, nxt, ans))
    print(h3(start, nxt, ans))
    print(h4(start, nxt, ans))
    print(h5(start, nxt, ans))
    print(h6(start, nxt, ans))
    #sample_questions('questions.txt')
    

if __name__ == '__main__':
    __main__()
