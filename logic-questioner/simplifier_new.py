from create_expressions_mistakes import *
from contextlib import redirect_stdout
import networkx as nx
import io

from astar import AStar

class AStarSolution(AStar):
    def neighbors(self, node):
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
        #node = convert_to_logic_symbols(node)
        try:
            trainer = LogicTreeTrainer(first_tree = node, expand=None, op_seq=ops, op_pairs=False)
        except:
            raise ValueError('Could not parse', node)

        trainer.increment_ops(num_ops)
        trees = trainer.get_trees()

        tree_strs = [t.parse_tree() for t in trees]
        return tree_strs

    def distance_between(self, n1, n2):
        return abs(len(current) - len(goal))

    def heuristic_cost_estimate(self, current, goal):
        return abs(len(current) - len(goal))

    def is_goal_reached(self, current, goal):
        return current == goal

def __main__():
    start_expr = input("Start Expression: ")
    end_expr = input("End Expression: ")
    problem = AStarSolution()
    solution = problem.astar(start_expr, end_expr)
    for step in solution:
        print(step)

if __name__ == '__main__':
    __main__()
