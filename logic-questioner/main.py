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
questions = ["Prove that (p∨q)∨(p∨~q) is a tautology.",
             "Prove that ((p→r)∧(q→r)∧(p∨q))→r is a tautology.",
             "Prove that (~(~p))↔p is a tautology.",
             "Prove that ((p→q)∧(q→r))→(p→r) is a tautology."][0:1]
answers = ["T", "T", "T", "T"][0:1]
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


class StepForm(FlaskForm):
    step = StringField(label="Step")
    law = SelectField(label="Law", choices=laws)
    error = None
    delete_button = SubmitField('X')


class WireForm(Form):
    question = Label(field_id=0, text=random.choice(questions))
    steps = FieldList(FormField(StepForm), min_entries=1)
    output = ""
    mode = RadioField('choice', validators=[DataRequired('Please select assessment mode!')],choices=[('practice', 'Practice'), ('test', 'Test')], default='practice')


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
            return redirect(url_for('solve', username=request.form['username'], password=request.form['password']))
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
    has_error = False

    if request.method == 'POST':
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
                    elif i != 0 and not check_correct_operation(form.steps[i-1].data['step'], step.data['step'], ops=[step.data['law']]*3, num_ops=3):
                        has_error = True
                        step.error = 'Did NOT apply %s correctly!' % step.data['law']
                    else:
                        step.error = None


            if not has_error and form.data['steps'][-1]['step'].strip() == "T":
                form.output = 'CORRECT!'

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
