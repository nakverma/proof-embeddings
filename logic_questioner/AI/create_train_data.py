import json
from lark import Token
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


def create_dist_data(question_list, train_file="training_data_dist.txt"):  # corpus of 'sentences' of next steps
    seen = set()

    def write_sample(q, tf, max_depth):
        val = lambda s: s.value if type(s) == Token else s
        if len(q) == 1:
            return
        for i, s1 in enumerate(q):
            for j, s2 in enumerate(q[i:]):
                out = str((val(s1[0]), val(s2[0]), abs(max_depth-j)/max_depth))
                if out not in seen:
                    seen.add(out)
                tf.write(out + "\n")

    qg = QuestionGenerator()
    print("Creating Corpus...")
    with open(train_file, "w") as tf:
        for q in question_list:
            for s in range(3):
                qp, qt = qg.generate(q['premise'], max_depth=5), qg.generate(q['target'], max_depth=5)
                qp, qt = qp["solution"], qt["solution"]
                write_sample(qp, tf, 5)
                write_sample(qt, tf, 5)
            print(".", end="")
        print()
    print("Dist Corpus Created")


def create_dist_data_small(question_list, train_file="training_data_dist_small.txt"):  # corpus of 'sentences' of next steps
    seen = set()

    def write_sample(q, tf, max_depth):
        val = lambda s: s.value if type(s) == Token else s
        if len(q) == 1:
            return
        for i, s1 in enumerate(q):
            for j, s2 in enumerate(q[i:]):
                out = str((val(s1[0]), val(s2[0]), abs(max_depth-j)/max_depth))
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
                write_sample(qp, tf, 5)
                write_sample(qt, tf, 5)
            print(".", end="")
        print()
    print("Dist Corpus Created")


def create_contrastive_data(question_list, train_file="training_data_contrastive.txt"):  # corpus of 'sentences' of next steps
    seen = set()

    def write_sample(q, tf, max_depth):
        val = lambda s: s.value if type(s) == Token else s
        if len(q) == 1:
            return
        for i, s1 in enumerate(q):
            for j, s2 in enumerate(q[i:]):
                out = str((val(s1[0]), val(s2[0]), 1 if abs(j-i) < 2 else 0))
                if out not in seen:
                    seen.add(out)
                tf.write(out + "\n")

    qg = QuestionGenerator()
    print("Creating Corpus...")
    with open(train_file, "w") as tf:
        for q in question_list[:5]:
            for s in range(3):
                qp, qt = qg.generate(q['premise'], max_depth=10), qg.generate(q['target'], max_depth=10)
                qp, qt = qp["solution"], qt["solution"]
                write_sample(qp, tf, 10)
                write_sample(qt, tf, 10)
            print(".", end="")
        print()
    print("Contrastive Corpus Created")

if __name__ == "__main__":
    ql = get_question_list()
    #create_corpus(ql)
    #create_rule_corpus(ql)
    #create_step_data(ql)
    #create_dist_data(ql)
    #create_dist_data_small(ql)
    create_contrastive_data(ql)