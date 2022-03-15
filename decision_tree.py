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

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


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


# if __name__ == '__main__':
#     training_data = load_csv('wine2.csv')
#     for row in training_data:
#         row = convertToFloat(row)
#     my_tree = build_tree(training_data)
    
#     testing_data = [
#         [13.75,1.73,2.41,16,89,2.6,2.76,.29,1.81,5.6,1.15,2.9,1320,1],
#         [13.39,1.77,2.62,16.1,93,2.85,2.94,.34,1.45,4.8,.92,3.22,1195,1],
#         [12.29,2.83,2.22,18,88,2.45,2.25,.25,1.99,2.15,1.15,3.3,290,2],
#         [11.45,2.4,2.42,20,96,2.9,2.79,.32,1.83,3.25,.8,3.39,625,2],
#         [14.13,4.1,2.74,24.5,96,2.05,.76,.56,1.35,9.2,.61,1.6,560,3],
#         [14.06,2.15,2.61,17.6,121,2.6,2.51,.31,1.25,5.05,1.06,3.58,1295,1],
#         [14.83,1.64,2.17,14,97,2.8,2.98,.29,1.98,5.2,1.08,2.85,1045,1],
#         [13.86,1.35,2.27,16,98,2.98,3.15,.22,1.85,7.22,1.01,3.55,1045,1],
#         [14.1,2.16,2.3,18,105,2.95,3.32,.22,2.38,5.75,1.25,3.17,1510,1],
#         [14.12,1.48,2.32,16.8,95,2.2,2.43,.26,1.57,5,1.17,2.82,1280,1],
#         [11.84,.89,2.58,18,94,2.2,2.21,.22,2.35,3.05,.79,3.08,520,2],
#         [12.67,.98,2.24,18,99,2.2,1.94,.3,1.46,2.62,1.23,3.16,450,2],
#         [12.16,1.61,2.31,22.8,90,1.78,1.69,.43,1.56,2.45,1.33,2.26,495,2],
#         [11.65,1.67,2.62,26,88,1.92,1.61,.4,1.34,2.6,1.36,3.21,562,2],
#         [11.64,2.06,2.46,21.6,84,1.95,1.69,.48,1.35,2.8,1,2.75,680,2],
#         [13.08,3.9,2.36,21.5,113,1.41,1.39,.34,1.14,9.40,.57,1.33,550,3],
#         [13.5,3.12,2.62,24,123,1.4,1.57,.22,1.25,8.60,.59,1.3,500,3],
#         [12.79,2.67,2.48,22,112,1.48,1.36,.24,1.26,10.8,.48,1.47,480,3],
#         [13.11,1.9,2.75,25.5,116,2.2,1.28,.26,1.56,7.1,.61,1.33,425,3],
#         [13.23,3.3,2.28,18.5,98,1.8,.83,.61,1.87,10.52,.56,1.51,675,3]
#     ]
#     print(len(training_data))
#     total_count = 0
#     matched = 0
#     for row in testing_data:
#         total_count += 1
#         print ("Actual: %s. Predicted: %s" %
#                (row[-1], print_leaf(classify(row, my_tree))))
#         classified = classify(row, my_tree)
#         for lbl in classified.keys():
#             if(lbl == row[-1]):
#                 matched += 1
#     accuracy = matched / total_count * 100.0
#     print("accuracy : ", accuracy, "%")
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





