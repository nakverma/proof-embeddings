from flask import Flask, render_template, request, redirect, url_for
from wtforms import Form, StringField, FormField, FieldList, Label, SelectField, SubmitField, RadioField
from wtforms.validators import DataRequired
from flask_wtf import FlaskForm


from check_syntax import checkSyntax, raw2latex, latex2raw
from deterministic import check_correct_operation
from create_expressions_mistakes import LogicTree

from datetime import datetime
import random

import boto3
import botocore
from botocore.exceptions import ClientError

app = Flask(__name__)
app.secret_key = "secret"

BUCKET_NAME = 'response-data' # replace with your bucket name
ANSWER_KEY = 'answer_data.csv' # replace with your object key
STEP_KEY = 'step_data.csv'

s3 = boto3.resource('s3')

# TODO: Slider hardcoded change that!
steps_init = [{"label": "Step 1"}, {"label": "Step 2"}, {"label": "Step 3"}][0:1]
completed_question = False

questions = [
    {'question': "Prove that (pvq)v(pv~q) is a tautology.",
     'answer': 'T',
     'difficulty': 'mild'},
    {'question': "Prove that ((p->r)^(q->r)^(pvq))->r is a tautology.",
     'answer': 'T',
     'difficulty': 'spicy'},
    {'question': "Prove that (~(~p))<->p is a tautology.",
     'answer': 'T',
     'difficulty': 'mild'},
    {'question': "Prove that ((p->q)^(q->r))->(p->r) is a tautology.",
     'answer': 'T',
     'difficulty': 'spicy'},
    {'question': "Prove that F->T is a tautology.",
     'answer': 'T',
     'difficulty': 'mild'},
    {'question': "Prove that (p^q)v(~pv(p^~q)) is a tautology.",
     'answer': 'T',
     'difficulty': 'medium'},
    {'question': "Prove that ((pvq)^(rv~q))->(pvr) is a tautology.",
     'answer': 'T',
     'difficulty': 'spicy'},
    {'question': "Prove that (p^q)->(pvq) is a tautology.",
     'answer': 'T',
     'difficulty': 'mild'},
    {'question': "Prove that q->(p->q) is a tautology.",
     'answer': 'T',
     'difficulty': 'mild'},
    {'question': "Prove that p^~(p^~q)<->p^q is a tautology.",
     'answer': 'T',
     'difficulty': 'spicy'},
    {'question': "Prove that ~(p->q)^(p^q^s->r)^p is a fallacy.",
     'answer': 'F',
     'difficulty': 'spicy'},
    {'question': "Prove that (~qvq)^~r^p^r is a fallacy.",
     'answer': 'F',
     'difficulty': 'medium'},
    {'question': "Prove that ~r^((~pvp)^r)^(p->r) is a fallacy.",
     'answer': 'F',
     'difficulty': 'medium'},
    {'question': "Prove that s^((~s^~q)v(~s^~T))^p is a fallacy.",
     'answer': 'F',
     'difficulty': 'medium'},
    {'question': "Prove that (p->q)^(q->r) is logically equivalent to p->(q^r).",
     'answer': 'p->(q^r)',
     'difficulty': 'medium'},
    {'question': "Prove that ~(~((q^r)v(q^~r))^p) is logically equivalent to p->q.",
     'answer': 'p->q',
     'difficulty': 'spicy'},
    {'question': "Prove that qv(p^~q) is logically equivalent to ~p->q.",
     'answer': '~p->q',
     'difficulty': 'mild'},
    {'question': "Prove that ~(~(((~p^s)v((~p^T)^~s))^p)^~p) is logically equivalent to p.",
     'answer': 'p',
     'difficulty': 'spicy'},
    {'question': "Prove that ~(q^~p)^(qv~p) is logically equivalent to p<->q.",
     'answer': 'p↔q',
     'difficulty': 'mild'},
    {'question': "Prove that ~(~r^~(~(p^(qvq)))) is logically equivalent to (p^q)->r.",
     'answer': '(p^q)->r',
     'difficulty': 'spicy'},
    {'question': "Prove that (p->q)->((p->q)->q) is logically equivalent to (pvq).",
     'answer': '(pvq)',
     'difficulty': 'medium'},
    {'question': "Prove that (pvq)^(pv~q) is logically equivalent to p.",
     'answer': 'p',
     'difficulty': 'medium'},
    {'question': "Prove that ~(p^~q)vq is logically equivalent to ~pvq.",
     'answer': '~pvq',
     'difficulty': 'mild'},
    {'question': "Prove that ~(p^q)^(pv~q) is logically equivalent to ~q.",
     'answer': '~q',
     'difficulty': 'mild'},
    {'question': "Prove that (pvq)^(~p->~q) is logically equivalent to p.",
     'answer': 'p',
     'difficulty': 'medium'}
]

questions_ = []
for question in questions:
    question['answer'] = raw2latex(question['answer'])
    q = "Prove that "
    q += raw2latex(question['question'].split('Prove that ')[-1].split(' is')[0])
    q += " is "
    q += " ".join(question['question'].split('Prove that ')[-1].split(' is')[-1].split()[:-1])
    q += " "
    last = question['question'].split('Prove that ')[-1].split(' is')[-1].split()[-1]
    if last != 'fallacy' and last != 'tautology':
        q += raw2latex(last)
    else:
        q += last
    question['question'] = q
    questions_.append(question)
questions = questions_

print("QUESTIONS: ", questions)

laws = list(LogicTree().op_optns_diict.keys())
laws.remove('ALL')
print('Using LAWS=', laws)


def step_input_check(step):
    if len(step.data['step'].strip()) == 0:
        return False
    return True


def step_law_check(step):
    if len(step.data['law']) == 0:
        return False
    return True


def step_syntax_check(step):
    if not checkSyntax(latex2raw(step.data['step'])):
        return False
    return True

def clean_step(step):
    cleaned_step = step.replace('‚à®', 'v')
    cleaned_step = cleaned_step.replace('‚àß', '^')
    cleaned_step = cleaned_step.replace('‚Üí', '->')
    cleaned_step = cleaned_step.replace('‚Üî', '<->')
    return cleaned_step

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
    mode = RadioField('choice',
                      validators=[DataRequired('Please select assessment mode!')],
                      choices=[('practice', 'Practice'), ('test', 'Test')],
                      default='practice')
    # NOTE: Default mode is made "practice" and the radio button option is removed visually
    #       (i.e. from HTML). This was suggested by Prof. Ansaf.
    difficulty = 'mild'
    showlaws = 0


@app.route('/', methods=['GET', 'POST'])
def main():
    # NOTE: For now, we are commenting out the login page because we're not collecting data.
    #       Later, we'll use it when we collect data.
    # return redirect(url_for('login'))
    question_text, question_answer = select_a_question('mild')
    return redirect(url_for('solve',
                            question_text=question_text,
                            question_answer=question_answer,
                            question_difficulty='mild',
                            showlaws=False))


"""
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
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
"""


@app.route('/solve', methods=['GET', 'POST'])
def solve():
    # NOTE: For now, we are commenting out the login page because we're not collecting data.
    #       Later, we'll use it when we collect data.
    """
    # TODO: Fix authentication! (Make sure username and password does not appear there!)
    try:
        username, password = request.args['username'], request.args['password']
        if not (username == 'admin' and password == 'admin'):
            return redirect(url_for('login'))
    except:
        return redirect(url_for('login'))
    """
    global completed_question

    form = WireForm(request.form, steps=steps_init)
    form.question.text = request.args['question_text']
    form.difficulty = request.args['question_difficulty']
    form.showlaws = request.args['showlaws']
    has_error = False

    req_ip = str(request.access_route[-1])
    usr_agent = str(request.user_agent.string).replace(",","")
    t = str(datetime.now())

    # TODO: Implement question difficulty and show/hide laws persistently!
    # TODO: There are some problems with the clear and delete button in terms of the visual
    #       persistent changes, investigate these!

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

        if "skip" in request.form or ("clear" not in request.form and "next" not in request.form and "end" not in request.form):
            if not completed_question:
                try:
                    s3.Bucket(BUCKET_NAME).download_file(ANSWER_KEY, 'local_answer_data.csv')
                except botocore.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == "404":
                        print("The object does not exist.")
                    else:
                        raise

                ans_data_csv = open('local_answer_data.csv', 'a')

                ans_data = req_ip+","+t+","+usr_agent+","
                ans_data += form.question.text + ",0,"

                if len(form.steps) == 1 and not form.steps[0].data['step']:
                    ans_data += "-1\n"
                else:
                    ans_data += str(len(form.steps) - 1) + "\n"


                ans_data_csv.write(ans_data)
                ans_data_csv.close()

                s3_client = boto3.client('s3')
                try:
                    response = s3_client.upload_file('local_answer_data.csv', BUCKET_NAME, ANSWER_KEY)
                except ClientError as e:
                    print(e)

            completed_question = False

            question_text, question_answer = select_a_question(request.form['difficulty'], current_question_text=request.args['question_text'])
            return redirect(url_for('solve',
                                    question_text=question_text,
                                    question_answer=question_answer,
                                    question_difficulty=request.form['difficulty'],
                                    showlaws=request.form['showlaws']))

        if "clear" in request.form:
            previous_data = form.data
            previous_data['steps'] = [{"step": "", "csrf_token": ""}]
            form.__init__(data=previous_data)
            form.showlaws = request.form['showlaws']
            return render_template("form.html", form=form)

        step_data = []
        # (IP, timestamp, question, step#, law, correct/incorrect)


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
            elif form.data['mode'] == 'practice' and i == 0 and not check_correct_operation(form.question.text.split('Prove that ')[-1].split(' is')[0], step.data['step'], ops=[step.data['law']], num_ops=3):
                has_error = True
                step.error = 'Did NOT apply %s correctly!' % step.data['law']

                step_data.append([req_ip, t, usr_agent, clean_step(form.question.text), i, step.data['law'], clean_step(step.data['step']), 0])

            elif form.data['mode'] == 'practice' and i != 0 and not check_correct_operation(form.steps[i-1].data['step'], step.data['step'], ops=[step.data['law']], num_ops=3):
                has_error = True
                step.error = 'Did NOT apply %s correctly!' % step.data['law']

                step_data.append([req_ip, t, usr_agent, clean_step(form.question.text), i, step.data['law'], clean_step(step.data['step']), 0])
            else:
                step.error = None

                step_data.append([req_ip, t, usr_agent, clean_step(form.question.text), i, step.data['law'], clean_step(step.data['step']), 1])

        if has_error:
            pass
        # elif "next" in request.form:
            # previous_data = form.data
            # previous_data['steps'].append({"step": "", "csrf_token": ""})
            # form.__init__(data=previous_data)
        # elif "end" in request.form:
        elif "next" in request.form:
            previous_data = form.data
            form.__init__(data=previous_data)
            form.showlaws = request.form['showlaws']

            # if form.data['mode'] == 'test':
            #     has_error = False
            #     for i, step in enumerate(form.steps):
            #         if not step_syntax_check(step):
            #             has_error = True
            #             step.error = 'Please use correct logic syntax in this step!'
            #         elif i == 0 and not check_correct_operation(form.question.text.split('Prove that ')[-1].split(' is')[0], step.data['step'], ops=[step.data['law']], num_ops=3):
            #             has_error = True
            #             step.error = 'Did NOT apply %s correctly!' % step.data['law']
            #
            #             if len(form.steps) == 1: # this is the only step
            #                 step_data = (form.question.text, i, step.data['law'], 0)
            #         elif i != 0:
            #             if not step_syntax_check(form.steps[i-1]):
            #                 has_error = True
            #                 step.error = 'Please use correct logic syntax in the previous step!'
            #             elif not check_correct_operation(form.steps[i-1].data['step'], step.data['step'], ops=[step.data['law']], num_ops=3):
            #                 has_error = True
            #                 step.error = 'Did NOT apply %s correctly!' % step.data['law']
            #
            #                 if i == len(form.steps)-1: # this is the most recent step
            #                     step_data = (form.question.text, i, step.data['law'], 0)
            #             else:
            #                 if form.data['mode'] == 'test' and i == len(form.steps)-1: # this is the most recent step
            #                     step_data = (form.question.text, i, step.data['law'], 1)
            #         else:
            #             step.error = None
            #
            #             if form.data['mode'] == 'test' and i == len(form.steps)-1: # this is the most recent step
            #                 step_data = (form.question.text, i, step.data['law'], 1)

            if not has_error and form.data['steps'][-1]['step'].strip() == request.args['question_answer']:
                form.output = 'CORRECT! Press "Next Question" to move on to the next question!'
                completed_question = True

                try:
                    s3.Bucket(BUCKET_NAME).download_file(ANSWER_KEY, 'local_answer_data.csv')
                except botocore.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == "404":
                        print("The object does not exist.")
                    else:
                        raise

                ans_data_csv = open('local_answer_data.csv', 'a')

                ans_data = req_ip+","+t+","+usr_agent+","
                ans_data += clean_step(form.question.text) + ",1," + str(len(form.steps) - 1) + "\n"


                ans_data_csv.write(ans_data)
                ans_data_csv.close()

                s3_client = boto3.client('s3')
                try:
                    response = s3_client.upload_file('local_answer_data.csv', BUCKET_NAME, ANSWER_KEY)
                except ClientError as e:
                    print(e)


            elif not has_error:
                previous_data = form.data
                previous_data['steps'].append({"step": "", "csrf_token": ""})
                form.__init__(data=previous_data)
                form.showlaws = request.form['showlaws']

        if step_data:
            step_commad = ""
            for entry in step_data:
                for item in entry:
                    step_commad += str(item) + ","
                step_commad += "\n"

            try:
                s3.Bucket(BUCKET_NAME).download_file(STEP_KEY, 'local_step_data.csv')
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    print("The object does not exist.")
                else:
                    raise

            step_data_csv = open('local_step_data.csv','a')  # 'a' option creates the file if not present, appends if present
            step_data_csv.write(step_commad)
            step_data_csv.close()

            s3_client = boto3.client('s3')
            try:
                response = s3_client.upload_file('local_step_data.csv', BUCKET_NAME, STEP_KEY)
            except ClientError as e:
                print(e)

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
