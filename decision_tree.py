from __future__ import print_function
from csv import reader
import math

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


def calculateEntropy(rows):
    counts = class_counts(rows)
    entropy = 0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        entropy -= math.log(prob_of_lbl,2)
    return entropy


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * calculateEntropy(left) - (1 - p) * calculateEntropy(right)


def find_best_split(rows):
    best_gain = 0 
    best_question = None 
    current_uncertainty = calculateEntropy(rows)
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
        [13.75,1.73,2.41,16,89,2.6,2.76,.29,1.81,5.6,1.15,2.9,1320,1],
        [13.39,1.77,2.62,16.1,93,2.85,2.94,.34,1.45,4.8,.92,3.22,1195,1],
        [12.29,2.83,2.22,18,88,2.45,2.25,.25,1.99,2.15,1.15,3.3,290,2],
        [11.45,2.4,2.42,20,96,2.9,2.79,.32,1.83,3.25,.8,3.39,625,2],
        [14.13,4.1,2.74,24.5,96,2.05,.76,.56,1.35,9.2,.61,1.6,560,3],
    ]

    total_count = 0
    matched = 0
    for row in testing_data:
        total_count += 1
        print ("Actual: %s. Predicted: %s" %
               (row[-1], print_leaf(classify(row, my_tree))))
        classified = classify(row, my_tree)
        for lbl in classified.keys():
            if(lbl == row[-1]):
                matched += 1
    accuracy = matched / total_count * 100.0
    print("accuracy : ", accuracy, "%")
