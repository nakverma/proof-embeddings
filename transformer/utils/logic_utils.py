import pickle
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd


class LogicLawCodes(object):
    IDENTITY = {
        'name': 'identity',
        'id': [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 19, 25, 27, 30, 31, 32, 33, 34, 35, 36],
        'label': 0
    }
    TAUTOLOGY = {
        'name': 'tautology',
        'id': [7, 8, 9],
        'label': 1
    }
    ASSOCIATIVITY = {
        'name': 'associativity',
        'id': [16, 17, 21, 22, 41, 42, 43, 44],
        'label': 2
    }
    COMMUTATIVITY = {
        'name': 'commutativity',
        'id': [15, 20],
        'label': 3
    }
    LOGICAL_EQUIVALENCE = {
        'name': 'logical_equivalence',
        'id': [23, 26],
        'label': 4
    }
    DEMORGAN = {
        'name': 'demorgan',
        'id': [18, 24, 28, 29],
        'label': 5
    }
    FALLACY = {
        'name': 'fallacy',
        'id': [37, 38, 39],
        'label': 6
    }

    LAWS = [IDENTITY, TAUTOLOGY, FALLACY, ASSOCIATIVITY, COMMUTATIVITY, LOGICAL_EQUIVALENCE, DEMORGAN]

    def label2name(self, label):
        for law in self.LAWS:
            if law['label'] == label:
                return law['name']

    def name2law(self, name):
        for law in self.LAWS:
            if law['name'] == name:
                return law


def load_pickle_data(data_path, specific_laws=None, data_mode='pairs', logging=True):
    """Function to load Michel's .pkl files which contains synthetic logic trees"""
    data = pickle.load(open(data_path, 'rb'))
    X, Y = [], []
    max_example_length, max_expression_length = 0, 0
    expression_lengths = []
    vocabulary_and_counts = defaultdict(int)

    # Specify which laws to pick from the data
    laws = []
    if not specific_laws:
        laws.extend(LogicLawCodes.LAWS)
    else:
        for name in specific_laws:
            laws.append(LogicLawCodes().name2law(name=name))

    for index, tup in tqdm(data.items(), desc='Loading Data from .pkl'):
        if len(tup[1]) <= 1:
            continue

        # Get the two, sequential expressions
        if data_mode == 'pairs':
            expression1, expression2 = np.array(tup[1][-1][0]), np.array(tup[1][-2][0])
            expressions = [str(expression1), str(expression2)]
            for law in laws:
                if tup[1][-1][1] in law['id']:
                    X.append(expressions)
                    Y.append(law['label'])
        # Get the full logic tree output, as a sequential list of expressions
        elif data_mode == 'sequential':
            expressions = []
            for i in range(len(tup[1])-1, -1, -1):
                expressions.append(str(np.array(tup[1][i][0])))
            X.append(expressions)
            # NOTE: We do this to preserve shape equality between X and Y; labels will NOT be
            #       utilized in the sequential scenario, therefore we use an arbitrary 99 here
            Y.append(99)
        else:
            raise ValueError('Data mode "%s" not recognized' % data_mode)

        # Update the max example length (i.e. how many expressions are in per example)
        # NOTE: We'll use it later in this function so that the final X array has well-defined shape
        max_example_length = max([max_example_length, len(expressions)])
        # Update the max expression length, which will be used to pad examples later on
        max_expression_length = max([max_expression_length, *[len(str(e)) for e in expressions]])
        for e in expressions:
            # Let's not count parentheses (i.e. '(' and ')') for computing distribution
            expression_lengths.append(len(str(e).replace('(', '').replace(')', '')))
            # Add tokens to vocabulary OR update counts
            for token in str(e):
                vocabulary_and_counts[token] += 1

    # Update examples lengths just in case an example can have differing num. expressions
    # NOTE: This is the case with 'sequential' data mode; a proof can have an arbitrary num. lines
    is_example_padded = False
    for i in range(len(X)):
        if len(X[i]) < max_example_length:
            X[i].extend([None] * (max_example_length - len(X[i])))
            if not is_example_padded:
                is_example_padded = True

    X, Y = np.array(X), np.array(Y)

    if logging:
        # Logs for sanity check
        print('Maximum Example Length: ', max_example_length)
        print('Examples are padded!' if is_example_padded else 'Examples are NOT padded!')
        print('Maximum (Single) Expression Char Length: ', max_expression_length)
        print('Num. Labels: ', len(np.unique(Y)))
        # Log distribution attributes
        print('MEAN: ', np.mean(expression_lengths))
        print('MIN: ', np.min(expression_lengths))
        print('MAX: ', np.max(expression_lengths))
        print('MEDIAN: ', np.median(expression_lengths))
        print('STD: ', np.std(expression_lengths))
        print('VOCABULARY AND COUNTS: ', vocabulary_and_counts)

    return X, Y


def load_tsv_data(data_path, data_mode='pairs', logging=True):
    """Function to load Dev's .tsv files which contains real student answers to logic questions"""
    data = pd.read_csv(data_path, sep='\t')
    if logging:
        print('Read TSV with columns: ', data.columns.tolist())

    X, Y = [], []
    max_example_length, max_expression_length = 0, 0
    expression_lengths = []
    vocabulary_and_counts = defaultdict(int)

    for _, row in tqdm(data.iterrows(), desc='Loading Data from .tsv'):
        expressions = []
        # Get the two, sequential expressions
        if data_mode == 'pairs':
            # If 'Student Response' column is empty, let's skip this row
            if pd.isna(row['Student Response']):
                continue
            # If 'Syntax Correction' column is not empty, let's replace 'Student Response' with it
            if pd.isna(row['Syntax Correction']):
                student_response = eval(row['Student Response'])
            else:
                student_response = eval(row['Syntax Correction'])

            # Zip expressions: zip([0, 1, 2], [1, 2]) -> [(0, 1), (1, 2)]
            expression_pairs = zip(student_response, student_response[1:])
            for index, pair in enumerate(expression_pairs):
                current_expressions = [ascii2common(str(pair[0])), ascii2common(str(pair[1]))]
                expressions.extend(current_expressions)
                # If the 'Error Steps' column has 0, it means the student has copied the question incorrectly, let's skip these
                if str(row['Error Steps']) == '0':
                    continue
                # If the 'Error Steps' column has i, j, ith pair indicates a wrong step
                elif not pd.isna(row['Error Steps']) and str(row['Error Steps']).split(',')[0] == str(index):
                    assert len(row['Error Steps'].split(',')) % 2 == 0
                    X.append(current_expressions)
                    Y.append(0)
                # If the 'Error Steps' column is empty or not equal to index, this step is correct
                elif pd.isna(row['Error Steps']):
                    X.append(current_expressions)
                    Y.append(1)
                else:
                    continue

        # Get the full logic tree output, as a sequential list of expressions
        elif data_mode == 'sequential':
            current_expressions = [ascii2common(expression) for expression in student_response]
            expressions.extend(current_expressions)
            X.append(current_expressions)
            # NOTE: We do this to preserve shape equality between X and Y; labels will NOT be
            #       utilized in the sequential scenario, therefore we use an arbitrary 99 here
            Y.append(99)
        else:
            raise ValueError('Data mode "%s" not recognized' % data_mode)

        # Update the max example length (i.e. how many expressions are in per example)
        # NOTE: We'll use it later in this function so that the final X array has well-defined shape
        max_example_length = max([max_example_length, len(expressions)])
        # Update the max expression length, which will be used to pad examples later on
        max_expression_length = max([max_expression_length, *[len(str(e)) for e in expressions]])
        for e in expressions:
            # Let's not count parentheses (i.e. '(' and ')') for computing distribution
            expression_lengths.append(len(str(e).replace('(', '').replace(')', '')))
            # Add tokens to vocabulary OR update counts
            for token in str(e):
                vocabulary_and_counts[token] += 1

    # Update examples lengths just in case an example can have differing num. expressions
    # NOTE: This is the case with 'sequential' data mode; a proof can have an arbitrary num. lines
    is_example_padded = False
    for i in range(len(X)):
        if len(X[i]) < max_example_length:
            X[i].extend([None] * (max_example_length - len(X[i])))
            if not is_example_padded:
                is_example_padded = True

    X, Y = np.array(X), np.array(Y)

    if logging:
        # Logs for sanity check
        print('Maximum Example Length: ', max_example_length)
        print('Examples are padded!' if is_example_padded else 'Examples are NOT padded!')
        print('Maximum (Single) Expression Char Length: ', max_expression_length)
        print('Num. Labels: ', len(np.unique(Y)))
        # Log distribution attributes
        print('MEAN: ', np.mean(expression_lengths))
        print('MIN: ', np.min(expression_lengths))
        print('MAX: ', np.max(expression_lengths))
        print('MEDIAN: ', np.median(expression_lengths))
        print('STD: ', np.std(expression_lengths))
        print('VOCABULARY AND COUNTS: ', vocabulary_and_counts)

    return X, Y


def ascii2common(expression):
    """
    Function to convert characters from real student data to the used vocabulary in synthetic
    data experiments. This will unify the vocabularies and help us combine the data. The difference
    in the vocabulary used arises because we wanted to use only ASCII characters in data annotation.
    """
    expression = expression.replace('v', '∨')
    expression = expression.replace('^', '∧')
    expression = expression.replace('->', '→')
    return expression
