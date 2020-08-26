from flask import Flask, render_template, request
from wtforms import Form, StringField, FormField, FieldList, Label, SelectField
from wtforms.validators import InputRequired
from flask_wtf import FlaskForm
import random

from check_syntax import checkSyntax

app = Flask(__name__)
app.secret_key = "secret"


steps_init = [{"label": "Step 1"}, {"label": "Step 2"}, {"label": "Step 3"}][0:1]
questions = ["Prove that (p∨q)∨(p∨¬q) is a tautology.",
             "Prove that ((p→r)∧(q→r)∧(p∨q))→r is a tautology.",
             "Prove that (¬(¬x))↔x is a tautology.",
             "Prove that ((p→q)∧(q→r))→(p→r) is a tautology."]
random_question = random.choice(questions)
laws = ['', 'Identity', 'Tautology', 'Associativity', 'Commutativity', 'Logical Equivalence', 'DeMorgan', 'Fallacy']


def step_input_check(step):
    if len(step.data['step'].strip()) == 0:
        return False
    return True


def step_syntax_check(step):
    if not checkSyntax(step.data['step']):
        return False
    return True


class StepForm(FlaskForm):
    step = StringField(label="Step")
    law = SelectField(label="Law", validators=[InputRequired()], choices=laws)
    error = None


class WireForm(Form):
    question = Label(field_id=0, text=random_question)
    steps = FieldList(FormField(StepForm), min_entries=1)
    output = ""


@app.route("/", methods=["GET", "POST"])
def wire_it():
    form = WireForm(request.form, steps=steps_init)
    has_error = False

    if request.method == "POST":
        for step in form.steps:
            if not step_input_check(step):
                has_error = True
                step.error = 'Please fill this step!'
            elif not step_syntax_check(step):
                has_error = True
                step.error = 'Please use correct logic syntax in this step!'
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
            form.output = steps2output(steps=previous_data["steps"])

    return render_template("form.html", form=form)


def steps2output(steps):
    output = []
    for step in steps:
        step_, law = step['step'].strip(), step['law']
        output.append((step_, law))

    return output


if __name__ == "__main__":
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True)
