from flask import Flask, render_template, request
from wtforms import Form, StringField, validators, FormField, FieldList, Label, TextAreaField, SelectField
from wtforms.validators import StopValidation, DataRequired
from flask_wtf import FlaskForm
import random
from wtforms.compat import string_types, text_type

from check_syntax import checkSyntax

app = Flask(__name__)
app.secret_key = "secret"


steps_init = [{"label": "Step 1"}, {"label": "Step 2"}, {"label": "Step 3"}][0:1]
questions = ["Prove that (p∨q)∨(p∨¬q) is a tautology.",
             "Prove that ((p→r)∧(q→r)∧(p∨q))→r is a tautology.",
             "Prove that (¬(¬x))↔x is a tautology.",
             "Prove that ((p→q)∧(q→r))→(p→r) is a tautology."]
random_question = random.choice(questions)
start_tokens = ['(', '~', '¬', 'p', 'q', 'r']


class InputRequired(object):
    """
    Validates that input was provided for this field.

    Note there is a distinction between this and DataRequired in that
    InputRequired looks that form-input data was provided, and DataRequired
    looks at the post-coercion data.
    """
    field_flags = ('required', )

    def __init__(self, message=None):
        self.message = message

    def __call__(self, form, field):
        if not field.raw_data or not field.raw_data[0]:
            if self.message is None:
                message = field.gettext('This field is required.')
            else:
                message = self.message

            field.errors[:] = []
            raise StopValidation(message)


class DataRequired(object):
    """
    Checks the field's data is 'truthy' otherwise stops the validation chain.

    This validator checks that the ``data`` attribute on the field is a 'true'
    value (effectively, it does ``if field.data``.) Furthermore, if the data
    is a string type, a string containing only whitespace characters is
    considered false.

    If the data is empty, also removes prior errors (such as processing errors)
    from the field.

    **NOTE** this validator used to be called `Required` but the way it behaved
    (requiring coerced data, not input data) meant it functioned in a way
    which was not symmetric to the `Optional` validator and furthermore caused
    confusion with certain fields which coerced data to 'falsey' values like
    ``0``, ``Decimal(0)``, ``time(0)`` etc. Unless a very specific reason
    exists, we recommend using the :class:`InputRequired` instead.

    :param message:
        Error message to raise in case of a validation error.
    """
    field_flags = ('required', )

    def __init__(self, message=None):
        self.message = message

    def __call__(self, form, field):
        print('ulaaan')
        if not field.data or isinstance(field.data, string_types) and not field.data.strip():
            if self.message is None:
                message = field.gettext('This field is required.')
            else:
                message = self.message

            field.errors[:] = []
            raise StopValidation(message)

class StepForm(FlaskForm):
    step = StringField(default="", label="Step", validators=[DataRequired()])
    options = start_tokens
    # TODO: Add a length validator in @validators above!


class WireForm(Form):
    question = Label(field_id=0, text=random_question)
    steps = FieldList(FormField(StepForm), min_entries=1)
    output = ""


@app.route("/", methods=["GET", "POST"])
def wire_it():
    form = WireForm(request.form, steps=steps_init)

    if request.method == "POST":
        if "next" in request.form:
            previous_data = form.data
            form.output = steps2output(steps=previous_data["steps"])
            # If the output is not well-formatted (i.e. undesirable character in user input),
            # we do not want to give the next step to the user.
            if 'wrong' in form.output:
                pass
            else:
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
        step_ = step['step'].strip()
        if checkSyntax(step_):
            output.append(step_)
        else:
            output.append('This is wrong!')

    return ' | '.join(output)


if __name__ == "__main__":
    app.run(debug=True)