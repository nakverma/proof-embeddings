from flask import Flask, render_template, request, redirect, url_for
from wtforms import Form, StringField, FormField, FieldList, Label, SelectField, SubmitField, RadioField
from wtforms.validators import InputRequired, DataRequired
from flask_wtf import FlaskForm
import random

from check_syntax import checkSyntax
from deterministic import check_correct_operation

app = Flask(__name__)
app.secret_key = "secret"

# TODO: Slider hardcoded change that!

steps_init = [{"label": "Step 1"}, {"label": "Step 2"}, {"label": "Step 3"}][0:1]

# TODO: Make sure the symbols above are correct e.g. ¬ instead of ~
"""
questions = ["Prove that (p∨q)∨(p∨~q) is a tautology.",
             "Prove that ((p→r)∧(q→r)∧(p∨q))→r is a tautology.",
             "Prove that (~(~p))↔p is a tautology.",
             "Prove that ((p→q)∧(q→r))→(p→r) is a tautology."]
answers = ["T", "T", "T", "T"][0:1]
"""
questions = [
    {'question': "Prove that (p∨q)∨(p∨~q) is a tautology.",
     'answer': 'T',
     'difficulty': 'mild'},
    {'question': "Prove that ((p→r)∧(q→r)∧(p∨q))→r is a tautology.",
     'answer': 'T',
     'difficulty': 'mild'},
    {'question': "Prove that (~(~p))↔p is a tautology.",
     'answer': 'T',
     'difficulty': 'mild'},
    {'question': "Prove that ((p→q)∧(q→r))→(p→r) is a tautology.",
     'answer': 'T',
     'difficulty': 'mild'},
    {'question': "Prove that F->T is a tautology.",
     'answer': 'T',
     'difficulty': 'mild'},
    {'question': "Prove that ~(p->q)^(p^q^s->r)^p is a fallacy.",
     'answer': 'F',
     'difficulty': 'medium'},
    {'question': "Prove that (~q∨q)∧~r∧p∧r is a fallacy.",
     'answer': 'F',
     'difficulty': 'medium'},
    {'question': "Prove that ~r∧((~p∨p)∧r)^(p->r) is a fallacy.",
     'answer': 'F',
     'difficulty': 'medium'},
    {'question': "Prove that s∧((~s∧~q)∨(~s∧~T))∧p is a fallacy.",
     'answer': 'F',
     'difficulty': 'medium'},
    {'question': "Prove that (p->q)^(q->r) is logically equivalent to p->(q^r).",
     'answer': 'p->(q^r)',
     'difficulty': 'spicy'},
    {'question': "Prove that ~(~((q∧r)∨(q∧~r))∧p) is logically equivalent to p->q.",
     'answer': 'p->q',
     'difficulty': 'spicy'},
    {'question': "Prove that q∨(p∧~q) is logically equivalent to ~p->q.",
     'answer': '~p->q',
     'difficulty': 'spicy'},
    {'question': "Prove that ~(~(((~p∧s)∨((~p∧T)∧~s))∧p)∧~p) is logically equivalent to p.",
     'answer': 'p',
     'difficulty': 'spicy'},
    {'question': "Prove that ~(q∧~p)∧(q∨~p) is logically equivalent to p↔q.",
     'answer': 'p↔q',
     'difficulty': 'spicy'},
    {'question': "Prove that ~(~r∧~(~(p∧(q∨q)))) is logically equivalent to (p^q)->r.",
     'answer': '(p^q)->r',
     'difficulty': 'spicy'},
]

laws = ['', 'IDENTITY', 'BOOLEAN_EQUIVALENCE', 'IMPLICATION_TO_DISJUNCTION', 'DOMINATION', 'IDEMPOTENCE', 
        'DOUBLE_NEGATION', 'COMMUTATIVITY', 'ASSOCIATIVITY', 'DISTRIBUTIVITY', 'NEGATION', 'DEMORGAN']


def step_input_check(step):
    if len(step.data['step'].strip()) == 0:
        return False
    return True


def step_law_check(step):
    if len(step.data['law']) == 0:
        return False
    return True


def step_syntax_check(step):
    if not checkSyntax(step.data['step']):
        return False
    return True

def select_a_question(difficulty='mild', current_question_text=None):
    questions_ = [question for question in questions if question['difficulty'] == difficulty and question['question'] != current_question_text]
    question = random.choice(questions_)
    question_text, question_answer = question['question'], question['answer']
    return question_text, question_answer

class StepForm(FlaskForm):
    step = StringField(label="Step")
    law = SelectField(label="Law", choices=laws)
    error = None
    delete_button = SubmitField('X')


class WireForm(Form):
    question = Label(field_id=0, text=random.choice(questions))
    steps = FieldList(FormField(StepForm), min_entries=1)
    output = ""
    mode = RadioField('choice', validators=[DataRequired('Please select assessment mode!')],choices=[('practice', 'Practice'), ('test', 'Test')], default='test')
    # NOTE: Default mode is made "test" and the radio button option is removed visually (i.e. from HTML). This was suggested by Prof. Ansaf.
    difficulty = 'mild'
    showlaws = 0


@app.route('/', methods=['GET', 'POST'])
def main():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            credentials_correct = False
            error = 'Invalid Credentials!'
        else:
            question_text, question_answer = select_a_question('mild')
            return redirect(url_for('solve', 
                                    username=request.form['username'], 
                                    password=request.form['password'],
                                    question_text=question_text,
                                    question_answer=question_answer,
                                    question_difficulty='mild',
                                    showlaws=False))
    return render_template('login.html', error=error)

@app.route('/solve', methods=['GET', 'POST'])
def solve():
    # TODO: Fix authentication! (Make sure username and password does not appear there!)
    try:
        username, password = request.args['username'], request.args['password']
        if not (username == 'admin' and password == 'admin'):
            return redirect(url_for('login'))
    except:
        return redirect(url_for('login'))

    form = WireForm(request.form, steps=steps_init)
    form.question.text = request.args['question_text']
    form.difficulty = request.args['question_difficulty']
    form.showlaws = request.args['showlaws']
    has_error = False
    # TODO: Implement question difficulty and show/hide laws persistently!
    # TODO: There are some problems with the clear and delete button in terms of the visual persistent changes, investigate these!

    if request.method == 'POST':
        if "skip" in request.form:
            question_text, question_answer = select_a_question(request.form['difficulty'], current_question_text=request.args['question_text'])
            return redirect(url_for('solve', 
                                    username=request.args['username'], 
                                    password=request.args['password'],
                                    question_text=question_text,
                                    question_answer=question_answer,
                                    question_difficulty=request.form['difficulty'],
                                    showlaws=request.form['showlaws'])) 

        elif "clear" in request.form:
            previous_data = form.data
            previous_data['steps'] = [{"step": "", "csrf_token": ""}]
            form.__init__(data=previous_data)
            return render_template("form.html", form=form)

        for i in range(len(form.steps)):
            if 'delete_%d' % (i+1) in request.form:
                previous_data = form.data
                del previous_data['steps'][i]
                if len(form.steps) == 1:
                    previous_data['steps'].append({"step": "", "csrf_token": ""})
                form.__init__(data=previous_data)
                return render_template("form.html", form=form)

        for i, step in enumerate(form.steps):
            if not step_input_check(step):
                has_error = True
                step.error = 'Please fill this step!'
            elif not step_law_check(step):
                has_error = True
                step.error = 'Please fill the law corresponding to this step!'
            elif form.data['mode'] == 'practice' and not step_syntax_check(step):
                has_error = True
                step.error = 'Please use correct logic syntax in this step!'
            elif form.data['mode'] == 'practice' and i == 0 and not check_correct_operation(form.question.text.split('Prove that ')[-1].split(' is')[0], step.data['step'], ops=[step.data['law']]*3, num_ops=3):
                has_error = True
                step.error = 'Did NOT apply %s correctly!' % step.data['law']
            elif form.data['mode'] == 'practice' and i != 0 and not check_correct_operation(form.steps[i-1].data['step'], step.data['step'], ops=[step.data['law']]*3, num_ops=3):
                has_error = True
                step.error = 'Did NOT apply %s correctly!' % step.data['law']
            else:
                step.error = None

        if has_error:
            pass
        elif "next" in request.form:
            previous_data = form.data
            previous_data['steps'].append({"step": "", "csrf_token": ""})
            form.__init__(data=previous_data)
        elif "end" in request.form:
            previous_data = form.data
            form.__init__(data=previous_data)

            if form.data['mode'] == 'test':
                has_error = False
                for i, step in enumerate(form.steps):
                    if not step_syntax_check(step):
                        has_error = True
                        step.error = 'Please use correct logic syntax in this step!'
                    elif i == 0 and not check_correct_operation(form.question.text.split('Prove that ')[-1].split(' is')[0], step.data['step'], ops=[step.data['law']]*3, num_ops=3):
                        has_error = True
                        step.error = 'Did NOT apply %s correctly!' % step.data['law']
                    elif i != 0:
                        if not step_syntax_check(form.steps[i-1]):
                            has_error = True 
                            step.error = 'Please use correct logic syntax in the previous step!'
                        elif not check_correct_operation(form.steps[i-1].data['step'], step.data['step'], ops=[step.data['law']]*3, num_ops=3):
                            has_error = True
                            step.error = 'Did NOT apply %s correctly!' % step.data['law']
                    else:
                        step.error = None


            if not has_error and form.data['steps'][-1]['step'].strip() == request.args['question_answer']:
                form.output = 'CORRECT! Press "Skip Question" to move on to the next question!' 

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
