
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

from Levenshtein._levenshtein import distance
import functools


#this is where the actual stuff for the algorithm is
def get_path(s1, s2):
    """ runs astar on the map"""

    def distance_between(n1, n2):
        return 1 # Only adjacent node pairs are ever considered

    def heuristic_cost_estimate(current, goal):
        tom = distance(current, goal)
        return tom

    def is_goal_reached(current, goal):
        return current == goal

    def convert_to_logic_symbols(expr):
        logic_symbols = ['∧', '∨', '→', '↔', '~']
        new_expr = expr.replace('^', '∧')
        new_expr = new_expr.replace('v', '∨')
        new_expr = new_expr.replace('<->', '↔')
        new_expr = new_expr.replace('->', '→')
        new_expr = new_expr.replace('x', 'p')
        new_expr = new_expr.replace('¬', '~')
        return new_expr

    def neighbors(node):
        next = []
        #same code as create_expression_mistakes main
        seed = node
        #print("----") #for debugging, to see each round
        trainer = LogicTreeTrainer(seed, expand=None)
        trainer.increment_ops(1)
        for key in trainer.trees.keys():
            if key > 1:
                next.append(str(trainer.trees[key][0]))
        return next


    s1 = convert_to_logic_symbols(s1)
    s2 = convert_to_logic_symbols(s2)

    #the distance, neighbors, and heuristic are all built in here
    return astar.find_path(s1, s2, neighbors_fnct=neighbors, heuristic_cost_estimate_fnct=heuristic_cost_estimate, distance_between_fnct=distance_between, is_goal_reached_fnct=is_goal_reached)



if __name__ == '__main__':

    # if len(sys.argv) != 3:
    #     print(
    #         'Usage : {script} <start_expression> <answer>'.format(script=sys.argv[0]))
    #     sys.exit(1)

    #station1 = get_station_by_name(sys.argv[1])
    #start = sys.argv[1]

    with open('questions.txt') as file:
        questions = json.load(file)
    print("Used Heuristic: abs(length of possible step - length of answer)")
    total_time = 0

    for i, question in enumerate(questions):
        try:
            start = question['question'].split('that ')[1].split(' is')[0]
            print('Start: ', start)
            ans = question['answer']
            print('Answer: ', ans)
            result_tuple = get_path(start, ans)
            print("------**-------")
            if result_tuple[0]:
                for s in result_tuple[0]:
                    print(s)
            else:
                raise Exception('path not found!')
            with open('results.txt','a') as file:
                file.write('Question {}: {} nodes considered. {} bailouts.\n'.format(i, result_tuple[1], result_tuple[2]))
                print('Question {}: {} nodes considered. {} bailouts.\n'.format(i, result_tuple[1], result_tuple[2]))
        except Exception as e:
            with open('results.txt','a') as file:
                file.write('Question {}: Error. {}\n'.format(i, e))
                print('Question {}: Error. {}\n'.format(i, e))

        print('\n*************************************\n')
