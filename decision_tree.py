from __future__ import print_function
from csv import reader
import math
from random import randint, seed

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

class Question:

    def __init__(self, index, value):
        self.index = index
        self.value = value

    def match(self, example):
        val = example[self.index]
        return val >= self.value

def partition(rows, question):
   
    right, left = [], []
    for row in rows:
        if question.match(row):
            right.append(row)
        else:
            left.append(row)
    return right, left


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
            right, left = partition(rows, question)
            if len(right) == 0 or len(left) == 0:
                continue
            gain = info_gain(right, left, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

class Leaf:
    
    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:

    def __init__(self,question,right,left):
        self.question = question
        self.right = right
        self.left = left


def build_tree(rows):
    
    gain, question = find_best_split(rows)
    if gain == 0:
        return Leaf(rows)

    right_rows, left_rows = partition(rows, question)
    right_branch = build_tree(right_rows)
    left_branch = build_tree(left_rows)
    return Decision_Node(question, right_branch, left_branch)


def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print (spacing + "Node class and count: ", node.predictions)
        return
    print (spacing + 'index: ' + str(node.question.index) + ' value: ' + str(node.question.value))

    print (spacing + '--> greater than:')
    print_tree(node.right, spacing + "  ")

    print (spacing + '--> less than:')
    print_tree(node.left, spacing + "  ")


def classify(row, node):
    
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.right)
    else:
        return classify(row, node.left)


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

seed(1)
if __name__ == '__main__':
    dataset = load_csv('wine.csv')
    total_accurary = 0.0
    for row in dataset:
        row = convertToFloat(row)
    groups = list()
    groupsize = int(len(dataset)/10)
    for i in range (10):
        group = list()
        for j in range (groupsize):
            idx = randint(0, len(dataset)-1)
            group.append(dataset.pop(idx))
        groups.append(group)
    for group in groups:
        trainData = list(groups) 
        trainData.remove(group)
        trainData = sum(trainData, [])
        testData = group
        my_tree = build_tree(trainData)
        print_tree(my_tree, "")
        total_row = 0
        total_matched = 0
        for row in testData:
            total_row += 1
            classified = classify(row,my_tree)
            for lbl in classified.keys():
                if lbl == row[-1]:
                    total_matched += 1
        accuracy = total_matched/total_row*100
        total_accurary += accuracy
        print('accurary: ', accuracy, '%')
    
    average_accuracy = total_accurary/ 10
    print('average accuracy: ', average_accuracy, "%")





