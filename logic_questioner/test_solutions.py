from expression_parser import get_frontier, validate
from validation_exception import InvalidExpressionException
import json


def question_solver(question):
    if question["premise"] != question["solution"][0]["statement"]:
        return f"Premise: {question['premise']} is not start: {question['solution'][0]['statement']}"
    if question["target"] != question["solution"][-1]["statement"]:
        return "Target is not end"
    for i, s in enumerate(question["solution"][:-1]):
        frontier = get_frontier(s["statement"])
        try:
            validate(frontier, question["solution"][i+1]["statement"], question["solution"][i+1]["rule"])
        except InvalidExpressionException:
            return f"Step {i+1}: {question['solution'][i+1]} not entailed by {s}"
    return "Passed"


def test_question_solver(question_file="questions.json"):
    with open(question_file, "r") as qf:
        questions = json.load(qf)['questions']
    fails = 0
    for i, q in enumerate(questions):
        res = question_solver(q)
        if res != "Passed":
            print(f"Question {i}: {res}")
            fails += 1
    print(f"Test Completed! Failures: {fails}")


if __name__ == "__main__":
    test_question_solver()
