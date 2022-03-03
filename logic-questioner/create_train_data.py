import json
from expression_parser import get_frontier
from question_generator import QuestionGenerator


def get_question_list(question_file="questions.json"):
    with open(question_file, "r") as qf:
        question_list = json.load(qf)['questions']
    return question_list


def create_corpus(question_list, train_file="training_data.txt"):  # corpus of 'sentences' of next steps
    qg = QuestionGenerator()
    print("Creating Corpus...")
    with open(train_file, "w") as tf:
        for q in question_list:
            for s in list(range(2, 6))*2:
                qp, qt = qg.generate(q['premise'], max_depth=s), qg.generate(q['target'], max_depth=s)
                qp, qt = qp["solution"], qt["solution"]
                if len(qp) != 1:
                    qp = ",".join([step[0] for step in qp])
                    tf.write(qp + "\n")
                if len(qt) != 1:
                    qt = ",".join([step[0] for step in qt])
                    tf.write(qt + "\n")
            print(".", end="")
        print()
    print("Corpus Created")


if __name__ == "__main__":
    ql = get_question_list()
    create_corpus(ql)

