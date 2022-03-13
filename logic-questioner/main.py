from flask import Flask, render_template, request, redirect, url_for
from wtforms import Form, StringField, FormField, FieldList, Label, SelectField, SubmitField, RadioField
from wtforms.validators import DataRequired
from flask_wtf import FlaskForm

import json

from datetime import datetime
import random
import string
import gc

from logic_rule_transforms import operation_names
from expression_parser import validate_and_get_frontier

import os


def create_session_id():
    length = 10
    random.seed(datetime.now())
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))


def raw2latex(s):
    s = s.replace("v", "V")
    s = s.replace("^", "∧")
    s = s.replace("<->", "↔")
    s = s.replace("->", "→")
    s = s.replace("~", "¬")
    return s


def latex2raw(s):
    s = s.replace("¬", "~")
    s = s.replace("→", "->")
    s = s.replace("↔", "<->")
    s = s.replace("∧", "^")
    s = s.replace("v", "V")
    return s


def get_laws():
    return list(operation_names.keys())


def get_question_list(filename='questions.json'):
    with open(filename, 'r') as qf:
        qs = json.load(qf)["questions"]
    for q in qs:
        answer = " is logically equivalent to " + raw2latex(q["target"]) + "."
        if q["target"] == "T":
            answer = " is a tautology."
        elif q["target"] == "F":
            answer = " is a fallacy."
        q["phrase"] = "Prove that " + raw2latex(q["premise"]) + answer
    return qs


def select_question(question_list, difficulty="mild", current_text=None):
    valid_qs = list(filter(lambda q: q["difficulty"] == difficulty and q["phrase"] != current_text, question_list))
    question = random.choice(valid_qs)
    return question["premise"], question["phrase"], question["target"], question["solution"]


app = Flask(__name__)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

questions = get_question_list()
laws = get_laws()
q_sol = None
completed_question = False
steps_init = {"label": "Step 1"}


class StepForm(FlaskForm):
    step = StringField(label="Step")
    law = SelectField(label="Law", choices=[""] + laws)
    error = None
    delete_button = SubmitField('X')


class WireForm(Form):
    question = Label(field_id=0, text=random.choice(questions))
    steps = FieldList(FormField(StepForm), min_entries=1)
    output = ""
    mode = RadioField('choice',
                      validators=[DataRequired('Please select assessment mode!')],
                      choices=[('practice', 'Practice'), ('test', 'Test')],
                      default='practice')
    difficulty = 'mild'
    showlaws = 0
    solution = []


@app.route('/', methods=['GET', 'POST'])
def main():
    premise, question_text, question_target, question_solution = select_question(questions, 'mild')
    global q_sol
    q_sol = {"premise": premise, "sol": question_solution, "target": question_target}

    return redirect(url_for(
        'solve',
        question_text=question_text,
        question_answer=question_target,
        question_difficulty='mild',
        showlaws=False,
        question_solution=json.dumps(q_sol),
        sid=create_session_id()
    ))


@app.route('/solve', methods=['GET', 'POST'])
def solve():
    global completed_question
    global q_sol

    form = WireForm(request.form, steps=steps_init)
    form.question.text = request.args['question_text']
    form.difficulty = request.args['question_difficulty']
    form.showlaws = request.args['showlaws']
    form.solution = request.args['question_solution']

    #print("\n", form.solution, "\n")
    q_sol = json.loads(form.solution)
    #print("\n", q_sol, "\n")
    
    has_error = False

    req_ip = str(request.access_route[-1])
    usr_agent = str(request.user_agent.string).replace(",", "")
    t = str(datetime.now())
    session_id = str(request.args['sid'])

    if request.method == 'POST':
        for i in range(len(form.steps)):
            if 'delete_%d' % (i + 1) in request.form:
                previous_data = form.data
                del previous_data['steps'][i]
                if len(form.steps) == 1:
                    previous_data['steps'].append({"step": "", "csrf_token": ""})
                form.__init__(data=previous_data)
                form.showlaws = request.form['showlaws']

                return render_template("form.html", form=form)

        if "skip" in request.form or ("clear" not in request.form and "next" not in request.form and "end" not in request.form and "getHint" not in request.form):
            completed_question = False
            premise, question_text, question_target, question_solution = select_question(
                questions, request.form['difficulty'], current_text=request.args['question_text'])

            q_sol = {"premise": premise, "sol": question_solution, "target": question_target}
            return redirect(url_for('solve',
                                    question_text=question_text,
                                    question_answer=question_target,
                                    question_difficulty=request.form['difficulty'],
                                    showlaws=request.form['showlaws'],
                                    question_solution=json.dumps(q_sol),
                                    sid=create_session_id()))

        if "clear" in request.form:
            previous_data = form.data
            previous_data['steps'] = [{"step": "", "csrf_token": ""}]
            form.__init__(data=previous_data)
            form.showlaws = request.form['showlaws']
            return render_template("form.html", form=form)

        if "getHint" in request.form:
            pass

        step_data = []
        for i, step in enumerate(form.steps):
            if i != len(form.steps) - 1:
                step.data["step"] = latex2raw(step.data["step"])
                step_data.append([req_ip, t, usr_agent, form.question.text, session_id, i, step.data['law'],  latex2raw(step.data["step"]), 1])
                continue
            prev_step = q_sol["premise"] if i == 0 else form.steps[i-1].data["step"]
            #print("\n", latex2raw(prev_step), latex2raw(step.data["step"]), step.data["law"], q_sol["sol"], "\n", sep="\n")
            print(f"{{\"statement\": \"{latex2raw(step.data['step'])}\", \"rule\": \"{step.data['law']}\"}}")
            check = validate_and_get_frontier(latex2raw(prev_step), latex2raw(step.data["step"]), step.data["law"], q_sol["target"])

            if not check["isValid"]:
                has_error = True
                step.error = check["errorMsg"]
            else:
                step.error = None
                step_data.append([req_ip, t, usr_agent, form.question.text, session_id, i, step.data['law'],  latex2raw(step.data["step"]), 1])

        gc.collect()

        if has_error:
            pass

        elif "next" in request.form:
            previous_data = form.data
            form.__init__(data=previous_data)

            completed_question = False
            if check["isSolution"]:
                form.output = 'CORRECT! Press "Next Question" to move on to the next question!'
                completed_question = True
            elif not has_error:
                previous_data = form.data
                previous_data['steps'].append({"step": "", "csrf_token": ""})
                form.__init__(data=previous_data)

        form.showlaws = request.form['showlaws']

    return render_template("form.html", form=form)


def steps2output(steps):
    output = []
    for step in steps:
        step_, law = step['step'].strip(), step['law']
        output.append((step_, law))

    return output


if __name__ == "__main__":
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)

print(select_question(get_question_list()))


