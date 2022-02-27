from astar import find_path
from expression_parser import *
from astar_heuristics import *
import json


def frontier_func(x):
    fr = get_frontier(x[0])
    # print(fr)
    return fr


def goal_func(x, target):
    # print(x, target)
    return x[0] == target


def get_goal_path(start, goal, goal_heuristic, adjacent_heuristic, frontier_getter=frontier_func, goal_check=goal_func):

    start = (start, 'Start')
    return find_path(
        start=start,
        goal=goal,
        heuristic_cost_estimate_fnct=goal_heuristic,
        distance_between_fnct=adjacent_heuristic,
        neighbors_fnct=frontier_getter,
        is_goal_reached_fnct=goal_check,
    )


if __name__ == "__main__":
    with open('questions.json') as f:
        questions = json.load(f)['questions']
    for q in questions[:1]:
        gp = get_goal_path(
            q['premise'], q['target'], levenshtein_distance, unitary_distance, frontier_func, goal_func
        )
        print("\n\n")
        if gp[0]:
            for step in gp[0]:
                print(step)
