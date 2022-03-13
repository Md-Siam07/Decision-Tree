from __future__ import print_function
from csv import reader

def load_csv(filename):
	file = open(filename, "rt")
	lines = reader(file)
	dataset = list(lines)
	return dataset

def convertToFloat(row):
	for i in range (0, len(row)):
            row[i] = float(row[i].strip())
		    

def unique_vals(rows_of_dataset, col):
    return set([row[col] for row in rows_of_dataset])

def class_counts(rows_of_dataset):
    counts = {}  
    for row in rows_of_dataset:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


class Question:

    def __init__(self, index, value):
        self.index = index
        self.value = value

    def match(self, example):
        val = example[self.index]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

def partition(rows, question):
   
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):
    best_gain = 0 
    best_question = None 
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  

    for col in range(n_features): 

        values = set([row[col] for row in rows]) 

        for val in values:  

            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

class Leaf:
    
    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    
    gain, question = find_best_split(rows)
    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return
    print (spacing + str(node.question))

    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def classify(row, node):
    
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


if __name__ == '__main__':
    training_data = load_csv('wine.csv')
    for row in training_data:
        row = convertToFloat(row)
    my_tree = build_tree(training_data)
    
    testing_data = [
        [13.73, 1.5, 2.7, 22.5, 101, 3, 3.25, 0.29, 2.38, 5.7, 1.19, 2.71, 1285, 1],
        [14.22, 1.7, 2.3, 16.3, 118, 3.2, 3, 0.26, 2.03, 6.38, 0.94, 3.31, 970, 1],
        [12.21, 1.19, 1.75, 16.8, 151, 1.85, 1.28, 0.14, 2.5, 2.85, 1.28, 3.07, 718, 2],
        [11.79, 2.13, 2.78, 28.5, 92, 2.13, 2.24, 0.58, 1.76, 3, 0.97, 2.44, 466, 2],
        [12.93, 2.81, 2.7, 21, 96, 1.54, 0.5, 0.53, 0.75, 4.6, 0.77, 2.31, 600, 3],
    ]

    for row in testing_data:
        print ("Actual: %s. Predicted: %s" %
               (row[-1], print_leaf(classify(row, my_tree))))
