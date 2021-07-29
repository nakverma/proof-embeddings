
import astar
import os
import sys
import csv
import math
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
        #print("RIGHT HERE" , str(n1))
        return abs(len(str(n1)) - len(str(n2)))

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

            for key in trainer.trees.keys(): #WORKING ON THIS PART
                if key > 1: #1 is just the input
                    #print(trainer.trees[key][1][1][1], " : ", trainer.trees[key][0])
                    #print(type(trainer.trees[key][0]))
                    next.append(str(trainer.trees[key][0]))

            #print(next)
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
    start = '(pvq)v(pv~q)'
    print('Start Expression : ' + start)
    #station2 = get_station_by_name(sys.argv[2])
    #ans = sys.argv[2]
    ans = 'T'
    print('Answer : ' + ans)
    #print('-' * 80)
    path = get_path(start, ans)
    print("------**-------")
    if path:
        for s in path:
            print(s)
    else:
        raise Exception('path not found!')