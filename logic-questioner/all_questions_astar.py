questions = open('questions.txt', 'r')
questions = json.load(questions)

def find_all_solutions:
    for question in questions:
        print("Queston:", question['question'])
        print("Answer":, question['answer'])
        solution = astar(question['question'], question['answer'])
        for step in solution:
            print(step)
