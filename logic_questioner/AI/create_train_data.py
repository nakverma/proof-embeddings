import json
from lark import Token
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from question_generator import QuestionGenerator


def get_question_list(question_file="../questions.json"):
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


def create_rule_corpus(question_list, train_file="training_data_rules.txt"):  # corpus of 'sentences' of next steps
    qg = QuestionGenerator()
    print("Creating Corpus...")
    with open(train_file, "w") as tf:
        for q in question_list:
            for s in list(range(2, 6))*2:
                qp, qt = qg.generate(q['premise'], max_depth=s), qg.generate(q['target'], max_depth=s)
                qp, qt = qp["solution"], qt["solution"]
                if len(qp) != 1:
                    qp = ",".join([str(step) for step in qp])
                    tf.write(qp + "\n")
                if len(qt) != 1:
                    qt = ",".join([str(step) for step in qt])
                    tf.write(qt + "\n")
            print(".", end="")
        print()
    print("Rule Corpus Created")


def create_step_data(question_list, train_file="training_data_steps.txt"):  # corpus of 'sentences' of next steps
    def write_sample(q, tf):
        val = lambda s: s.value if type(s) == Token else s
        if len(q) == 1:
            return
        target = q[-1][0]
        for i, s1 in enumerate(q):
            for j, s2 in enumerate(q[i:]):
                tf.write(
                    str((val(s1[0]), val(s2[0]), target, j, len(q)-1-i, len(q)-1-j-i)) + "\n"
                )  # e_n-e_c, e_t-e_c, e_t-e_n

    qg = QuestionGenerator()
    print("Creating Corpus...")
    with open(train_file, "w") as tf:
        for q in question_list:
            for s in list(range(2, 10)):
                qp, qt = qg.generate(q['premise'], max_depth=s), qg.generate(q['target'], max_depth=s)
                qp, qt = qp["solution"], qt["solution"]
                write_sample(qp, tf)
                write_sample(qt, tf)
            print(".", end="")
        print()
    print("Step Corpus Created")


def create_dist_data(question_list, train_file="train_data/training_data_dist.txt"):  # corpus of 'sentences' of next steps
    seen = set()

    def write_sample(q, tf):
        val = lambda s: s.value if type(s) == Token else s
        if len(q) == 1:
            return
        for i, s1 in enumerate(q):
            for j, s2 in enumerate(q[i:]):
                out1 = ",".join([val(s1[0]), val(s2[0]), str(j)])
                out2 = ",".join([val(s2[0]), val(s1[0]), str(j)])
                for out in (out1, out2):
                    if out not in seen:
                        seen.add(out)
                        tf.write(out + "\n")

    qg = QuestionGenerator()
    print("Creating Corpus...")
    with open(train_file, "w") as tf:
        for q in question_list:
            for s in range(3):
                qp, qt = qg.generate(q['premise'], max_depth=10), qg.generate(q['target'], max_depth=10)
                qp, qt = qp["solution"], qt["solution"]
                write_sample(qp, tf)
                write_sample(qt, tf)
            print(".", end="")
        print()
    print("Dist Corpus Created")


def create_dist_data_small(question_list, train_file="train_data/training_data_dist_small.txt"):  # corpus of 'sentences' of next steps
    seen = set()

    def write_sample(q, tf):
        val = lambda s: s.value if type(s) == Token else s
        if len(q) == 1:
            return
        for i, s1 in enumerate(q):
            for j, s2 in enumerate(q[i:]):
                out1 = ",".join([val(s1[0]), val(s2[0]), str(j)])
                out2 = ",".join([val(s2[0]), val(s1[0]), str(j)])
                for out in (out1, out2):
                    if out not in seen:
                        seen.add(out)
                        tf.write(out + "\n")

    qg = QuestionGenerator()
    print("Creating Corpus...")
    with open(train_file, "w") as tf:
        for q in question_list[:1]:
            for s in range(100):
                qp, qt = qg.generate(q['premise'], max_depth=5), qg.generate(q['target'], max_depth=5)
                qp, qt = qp["solution"], qt["solution"]
                write_sample(qp, tf)
                write_sample(qt, tf)
            print(".", end="")
        print()
    print("Dist Corpus Created")


def create_contrastive_data(question_list, window=3, train_file="train_data/training_data_contrastive_less.txt"):  # corpus of 'sentences' of next steps
    seen = set()

    def write_sample(q, tf):
        val = lambda s: s.value if type(s) == Token else s
        if len(q) == 1:
            return
        for i, s1 in enumerate(q):
            for j, s2 in enumerate(q[i:]):
                out1 = ",".join((val(s1[0]), val(s2[0]), "1" if j < window else "0"))
                out2 = ",".join((val(s2[0]), val(s1[0]), "1" if j < window else "0"))
                for out in [out1, out2]:
                    if out not in seen:
                        seen.add(out)
                        tf.write(out + "\n")

    qg = QuestionGenerator()
    print("Creating Corpus...")
    with open(train_file, "w") as tf:
        for q in question_list:
            for s in range(5):
                qp, qt = qg.generate(q['premise'], max_depth=7), qg.generate(q['target'], max_depth=10)
                qp, qt = qp["solution"], qt["solution"]
                write_sample(qp, tf)
                write_sample(qt, tf)
            print(".", end="")
        print()
    print("Contrastive Corpus Created")


def create_rule_data(question_list, train_file="train_data/training_data_rule.txt"):  # a, b, rule for a->b
    seen = set()

    def write_sample(q, tf):
        val = lambda s: s.value if type(s) == Token else s
        if len(q) == 1:
            return
        for i, s1 in enumerate(q[:-1]):
            s2 = q[i+1]
            out = ",".join((val(s1[0]), val(s2[0]), s2[1]))
            if out not in seen:
                seen.add(out)
                tf.write(out + "\n")

    qg = QuestionGenerator()
    print("Creating Corpus...")
    with open(train_file, "w") as tf:
        for q in question_list:
            for s in range(3):
                qp, qt = qg.generate(q['premise'], max_depth=7), qg.generate(q['target'], max_depth=10)
                qp, qt = qp["solution"], qt["solution"]
                write_sample(qp, tf)
                write_sample(qt, tf)
            print(".", end="")
        print()
    print("Rule Corpus Created")


def create_sol_contrastive(question_list, window=1, train_file="train_data/training_data_sol_contrastive.txt"):  # corpus of 'sentences' of next steps
    seen = set()

    def write_sample(q, tf):
        val = lambda s: s.value if type(s) == Token else s
        if len(q) == 1:
            return
        for i, s1 in enumerate(q):
            for j, s2 in enumerate(q[i:]):
                out1 = ",".join((val(s1[0]), val(s2[0]), "1" if j <= window else "0"))
                out2 = ",".join((val(s2[0]), val(s1[0]), "1" if j <= window else "0"))
                for out in [out1, out2]:
                    if out not in seen:
                        seen.add(out)
                        tf.write(out + "\n")

    print("Creating Corpus...")
    with open(train_file, "w") as tf:
        for q in question_list:
            qs = [(s["statement"], s["rule"]) for s in q["solution"]]
            write_sample(qs, tf)
            print(".", end="")
        print()
    print("Contrastive Sol Corpus Created")


if __name__ == "__main__":
    ql = get_question_list()
    #create_corpus(ql)
    #create_rule_corpus(ql)
    #create_step_data(ql)
    #create_dist_data(ql)
    create_dist_data_small(ql)
    #create_rule_data(ql)
    #create_contrastive_data(ql)
    #create_sol_contrastive(ql)