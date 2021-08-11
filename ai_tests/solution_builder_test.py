
import astar
import sys
sys.path.append("../logic_questioner") # Adds higher directory to python modules path
import os
import sys
import csv
import math
from difflib import SequenceMatcher
import unittest
#from ..logic_questioner.create_expressions_mistakes import LogicTreeTrainer
from create_expressions_mistakes import LogicTreeTrainer

# class Station:

#     def __init__(self, id, name, position):
#         self.id = id
#         self.name = name
#         self.position = position
#         self.links = []


# def build_data():
#     """builds the 'map' by reading the data files"""
#     stations = {}
#     rootdir = os.path.dirname(__file__)
#     r = csv.reader(open(os.path.join(rootdir, 'underground_stations.csv')))
#     next(r)  # jump the first line
#     for record in r:
#         id = int(record[0])
#         lat = float(record[1])
#         lon = float(record[2])
#         name = record[3]
#         stations[id] = Station(id, name, (lat, lon))

#     r = csv.reader(open(os.path.join(rootdir, 'underground_routes.csv')))
#     next(r)  # jump the first line
#     for id1, id2, lineNumber in r:
#         id1 = int(id1)
#         id2 = int(id2)
#         stations[id1].links.append(stations[id2])
#         stations[id2].links.append(stations[id1])
#     return stations

# STATIONS = build_data()

# def get_station_by_name(name):
#     """lookup by name, the name does not have to be exact."""
#     name = name.lower()
#     ratios = [(SequenceMatcher(None, name, v.name.lower()).ratio(), v)
#               for v in STATIONS.values()]
#     best = max(ratios, key=lambda a: a[0])
#     if best[0] > 0.7:
#         return best[1]
#     else:
#         return None


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


    def neighbors(node):
        def convert_to_logic_symbols(expr):
            logic_symbols = ['∧', '∨', '→', '↔', '~']
            new_expr = expr.replace('^', '∧')
            new_expr = new_expr.replace('v', '∨')
            new_expr = new_expr.replace('<->', '↔')
            new_expr = new_expr.replace('->', '→')
            new_expr = new_expr.replace('x', 'p')
            new_expr = new_expr.replace('¬', '~')
            return new_expr
        node = node.replace(' ', '')
        node = convert_to_logic_symbols(node)
        try:
            trainer = LogicTreeTrainer(node, expand=None, op_seq=ops, op_pairs=False)
        except:
            raise ValueError('Could not parse', start_expr)
        trainer.increment_ops(num_ops)
        trees = trainer.get_trees()
        tree_strs = [t.parse_tree() for t in trees]
        return tree_strs

    # s1 = convert_to_logic_symbols(s1)
    # s2 = convert_to_logic_symbols(s2)

    def distance_between(n1, n2):
        return math.abs(len(n1) - len(n2))

    def heuristic_cost_estimate(current, goal):
        return math.abs(len(current) - len(goal))

    def is_goal_reached(current, goal):
        return current == goal

    # def neighbors(node):
    #         #same code as create_expression_mistakes main
    #         seed = node
    #         trainer = LogicTreeTrainer(seed, expand=None)
    #         trainer.increment_ops(1)

    #         for key in treesMade.keys(): #WORKING ON THIS PART

    #             if key > 1: #1 is just the input
    #             print(treesMade[key][1][1][1], " : ", treesMade[key][0])




    #the distance, neighbors, and heuristic are all built in here
    return astar.find_path(s1, s2, neighbors_fnct=neighbors, heuristic_cost_estimate_fnct=heuristic_cost_estimate, distance_between_fnct=distance_between)


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

    if len(sys.argv) != 3:
        print(
            'Usage : {script} <start_expression> <answer>'.format(script=sys.argv[0]))
        sys.exit(1)

    #station1 = get_station_by_name(sys.argv[1])
    print('Start Expression : ' + sys.argv[1])
    #station2 = get_station_by_name(sys.argv[2])
    print('Answer : ' + sys.argv[2])
    print('-' * 80)
    path = get_path(station1, station2)
    if path:
        for s in path:
            print(s.name)
    else:
        raise Exception('path not found!')
