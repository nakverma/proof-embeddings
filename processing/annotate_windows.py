import os
import platform
import pandas as pd
from tqdm import tqdm
import argparse
import subprocess

SYSTEM = platform.system()
if SYSTEM.lower() != 'windows':
    raise ValueError('This script is only intended for Windows for now...')

# Get user inputs
parser = argparse.ArgumentParser(description='Script to annotate student answers')
parser.add_argument('--root_dir', help='The path to the root data directory containing PDF files', type=str, required=True)
parser.add_argument('--done_path', help='Path to .txt containing completely annotated files', type=str, required=True)
args = parser.parse_args()

# Sanity checks for user inputs
assert os.path.isdir(args.root_dir)
assert args.done_path.endswith('.txt')
if not os.path.exists(args.done_path):
    f = open(args.done_path, 'w')
    done = []
else:
    with open(args.done_path, 'r') as f:
        done =  [line.rstrip() for line in f if line.rstrip() != '']

print('COMPLETED FILES: ', done)

# Get semester from the user
print('Root contains the following semester folders: ', os.listdir(args.root_dir))
semester = input('Type a semester: ')
assert semester in os.listdir(args.root_dir)
semester_path = os.path.join(args.root_dir, semester)

# Get homework no. from the user
print('Semester contains the following homework folders: ', os.listdir(semester_path))
homework = input('Type a homework: ')
assert homework in os.listdir(semester_path)
homework_path = os.path.join(semester_path, homework)

# Get question & part from the user
question = input('Type question no.:')
part = input('Type part no.:')
assert question.isdigit() and part.isdigit()

print('The question ID is: ', '_'.join([semester, homework, 'Q%s' % question, 'P%s' % part]))
print('You might need to press [ENTER] occasionally if nothing is showing up...')
print('------------------------------------------------------------------------')

for submission_filename in tqdm(os.listdir(homework_path), desc='Annotating Student Submissions'):
    if not submission_filename.endswith('.pdf'):
        print('Skipping submission file [%s] with extension other than .pdf' % submission_filename)
        continue

    # Get the path to the submission file
    submission_filepath = os.path.join(homework_path, submission_filename)
    if submission_filepath in done:
        print('Skipping submission file [%s] since it is already annotated' % submission_filename)
        continue
    
    # Flash the PDF it on the screen
    try:
        subprocess.call('start %s' % submission_filepath, shell=True, stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT)
    except:
        print('Could not parse submission file [%s]' % submission_filename)
        continue

    # Get next operation from the user
    op = input('Press [n/N] for next file if the current file is annotated or [q/Q] to exit the program: ')
    if op.lower() == 'n':
        done.append(submission_filepath)
        continue
    elif op.lower() == 'q':
        break

# Save the done list into the .txt specified
with open(args.done_path, 'w') as f:
    for item in done:
        f.write('%s\n' % item)