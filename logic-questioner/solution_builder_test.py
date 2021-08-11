
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


#this is where the actual stuff for the algorithm is
def get_path(s1, s2):
    """ runs astar on the map"""


    # def distance_between(self, n1, n2):
    #     """this method always returns 1, as two 'neighbors' are always adajcent"""
    #     #took this from maze, this method (g) should be edited to have weights depending on the law used
    #     return 1

    # def distance(n1, n2):
    #     """computes the distance between two stations"""
    #     latA, longA = n1.position
    #     latB, longB = n2.position
    #     # convert degres to radians!!
    #     latA, latB, longA, longB = map(
    #         lambda d: d * math.pi / 180, (latA, latB, longA, longB))
    #     x = (longB - longA) * math.cos((latA + latB) / 2)
    #     y = latB - latA
    #     return math.hypot(x, y)


    # def neighbors(node):
    #     def convert_to_logic_symbols(expr):
    #         logic_symbols = ['∧', '∨', '→', '↔', '~']
    #         new_expr = expr.replace('^', '∧')
    #         new_expr = new_expr.replace('v', '∨')
    #         new_expr = new_expr.replace('<->', '↔')
    #         new_expr = new_expr.replace('->', '→')
    #         new_expr = new_expr.replace('x', 'p')
    #         new_expr = new_expr.replace('¬', '~')
    #         return new_expr
    #     node = node.replace(' ', '')
    #     node = convert_to_logic_symbols(node)
    #     try:
    #         trainer = LogicTreeTrainer(node, expand=None, op_seq=ops, op_pairs=False)
    #     except:
    #         raise ValueError('Could not parse', start_expr)
    #     trainer.increment_ops(num_ops)
    #     trees = trainer.get_trees()
    #     tree_strs = [t.parse_tree() for t in trees]
    #     return tree_strs



    def distance_between(n1, n2):
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

    def heuristic_cost_estimate(current, goal):
        return abs(len(str(current)) - len(str(goal)))

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

            #print("Right here " , trainer.trees[key][0])

            for key in trainer.trees.keys():
                if key > 1:
                    next.append(str(trainer.trees[key][0]))

            return next


    s1 = convert_to_logic_symbols(s1)
    s2 = convert_to_logic_symbols(s2)

    #the distance, neighbors, and heuristic are all built in here
    return astar.find_path(s1, s2, neighbors_fnct=neighbors, heuristic_cost_estimate_fnct=heuristic_cost_estimate, distance_between_fnct=distance_between, is_goal_reached_fnct=is_goal_reached)


# class LondonTests(unittest.TestCase):
#     def test_solve_underground(self):
#         for n1,n2 in [('Chesham', 'Wimbledon'), ('Uxbridge','Upminster'), ('Heathrow Terminal 4','Epping')]:
#             s1 = get_station_by_name(n1)
#             s2 = get_station_by_name(n2)
#             path = get_path(s1, s2)
#             self.assertTrue(not path is None)

# if __name__ == '__main__':
# 	print(solve_maze())

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
        start = question['question'].split('that ')[1].split(' is')[0]
        print('Start: ', start)
        ans = question['answer']
        print('Answer: ', ans)
        start_time = time.time()
        path = get_path(start, ans)
        end_time = time.time()
        completion_time = end_time - start_time
        print("------**-------")
        if path:
            for s in path:
                print(s)
        else:
            raise Exception('path not found!')

        print('Question ' + str(i+1) + ' Time:\t' + str(completion_time))
        total_time += completion_time
        print('\n*************************************\n')
    print('Total Completion Time: ' + str(total_time))


#
#            print("------**-------")
#            if path:
#                for s in path:
#                    print(s)
#            else:
#                raise Exception('path not found!')
#            print('\n*************************************\n')
#
