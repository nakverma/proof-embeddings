from astar import find_path
from expression_parser import *
from astar_heuristics import *
import json


def get_goal_path(start, goal, goal_heuristic, adjacent_heuristic, frontier_getter, goal_check):

    start, goal = (start, None), (goal, None)
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
        questions = json.load(f)
    for q in questions[:1]:
        start = q['question'].split('that ')[1].split(' is')[0]
        goal = q['answer']
        gp = get_goal_path(
            start, goal, levenshtein_distance, unitary_distance, lambda x: get_frontier(x[0]), lambda x, y: x == y
        )
        print("\n\n")
        if gp[0]:
            for step in gp[0]:
                print(step)

"""
OLD (to replicate logging)

if __name__ == '__main__':

    # if len(sys.argv) != 3:
    #     print(
    #         'Usage : {script} <start_expression> <answer>'.format(script=sys.argv[0]))
    #     sys.exit(1)

    #station1 = get_station_by_name(sys.argv[1])
    #start = sys.argv[1]

    with open('questions.json') as file:
        questions = json.load(file)
    print("Used Heuristic: abs(length of possible step - length of answer)")
    total_time = 0

    for i, question in enumerate(questions[:1]):
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
"""