---
layout: post
title: Fundamental Algorithms
subtitle: Decision Tree
tag: Machine Learning
comments:True
---
### Import Packages
~~~
import pandas as pd
from scipy.io import arff
from collections import Counter
import numpy as np
import sys
~~~

### Tree Construction
~~~
class Node:
    def __init__(self, parent):
        self.parent = parent
        self.children = []
        self.feature = None
        self.split = None
        self.label = None

~~~

### Nominal Features Split
~~~
def NominalSplit(Features, D, X):
    H = Entropy(Counter(D['class']))
    h, p = [], []
    for x in Features[X][1]:
        df = D[D[X].str.decode('utf-8') == x]
        h.append(Entropy(Counter(df['class'])))
        p.append(len(df) / len(D[X]))

    entropy = []
    for i in range(len(h)):
        entropy.append(p[i] * h[i])

    infogain = H - np.sum(entropy)
    return infogain
~~~

### Determine Candidate Splits
~~~
def DetermineCandidateSplits(D, X):
    Vbar = []
    InfoGain = []
    setdx = sorted(set(D[X]))
    if len(setdx) == 1:
        return 0, 0
    else:
        for i in range(len(setdx) - 1):
            vbar = (setdx[i] + setdx[i + 1]) / 2
            Vbar.append(vbar)
            C = [D[D[X] <= vbar], D[D[X] > vbar]]
            InfoGain.append(FindBestSplit(D, C))

        # break tie within one feature
        LargestestInfoGain = np.argwhere(InfoGain == np.amax(InfoGain)).flatten().tolist()
        Threshold = min([Vbar[x] for x in LargestestInfoGain])
        maxinfogain = max(InfoGain)

        return Threshold, maxinfogain


~~~
### Calculate Entropy
~~~
def Entropy(counts):
    try:
        Sum = sum((counts).values())
        p = [x / Sum for x in counts.values()]
        H = -p[0] * np.log2(p[0]) - p[1] * np.log2(p[1])
    except IndexError:
        H = 0
    return H

~~~
### Calculate Information Gain
~~~
def FindBestSplit(D, C):
    H = Entropy(Counter(D['class']))
    c1 = C[0]['class']
    c2 = C[1]['class']
    H1 = Entropy(Counter(c1))
    H2 = Entropy(Counter(c2))

    a = len(c1) / (len(c1) + len(c2))
    b = len(c2) / (len(c1) + len(c2))
    InfoGain = H - (a * H1 + b * H2)
    return InfoGain

~~~
### Core Algorithm
~~~

def MakeSubtree(root, D, Features, m):
    features = Features._attrnames[:-1]
    if len(D) >= m and len(Counter(D['class'])) > 1:
        InfoGain, Splits = [], []
        for X in features:
            if Features[X][0] == 'numeric':
                threshold, infogain = DetermineCandidateSplits(D, X)
                InfoGain.append(infogain)
                Splits.append(threshold)
            else:
                InfoGain.append(NominalSplit(Features, D, X))
                Splits.append(Features[X][1])

        # break tie between two features
        BestInfo = np.argwhere(InfoGain == np.amax(InfoGain)).flatten().tolist()

        feature = features[BestInfo[0]]
        split = Splits[BestInfo[0]]

        if max(InfoGain) == 0:
            root.feature = None
            root.split = None
            root.label = Counter(D['class'])
            return

        else:
            root.feature = feature
            root.label = Counter(D['class'])
            root.split = split
            if Features[feature][0] == 'numeric':
                root.children.append(Node(root))
                MakeSubtree(root.children[-1], D[D[feature] <= split], Features, m)
                root.children.append(Node(root))
                MakeSubtree(root.children[-1], D[D[feature] > split], Features, m)
            else:
                for value in split:
                    root.children.append(Node(root))
                    MakeSubtree(root.children[-1], D[D[feature].str.decode('utf-8') == value], Features, m)


    else:
        if D.empty:
            root.feature = None
            root.split = None
            root.label = {b'positive': 0, b'negative': 0}
            return
        else:
            root.feature = None
            root.split = None
            root.label = Counter(D['class'])
        return

~~~
### Determine Class
~~~
def DetermineClass(root):
    try:
        a = list(root.label.items())[0]
        b = list(root.label.items())[1]
        if a[1] > b[1]:
            return a[0].decode("utf-8")
        elif a[1] == b[1]:
            return DetermineClass(root.parent)
        else:
            return b[0].decode("utf-8")
    except IndexError:
        return list(root.label.keys())[0].decode("utf-8")

    except AttributeError:
        return 'negative'

~~~
### Prediction
~~~
def ClassPrediction(TestRow, root):
    if root.children:
        if type(root.split) == float:
            if TestRow[root.feature] <= root.split:
                return ClassPrediction(TestRow, root.children[0])
            else:
                return ClassPrediction(TestRow, root.children[1])
        else:
            for i in range(len(root.children)):
                if TestRow[root.feature].decode('utf-8') == root.split[i]:
                    return ClassPrediction(TestRow, root.children[i])
            return DetermineClass(root)
    else:
        return DetermineClass(root)
~~~
### Print Preorder Tree
~~~
def PrintNumericNodes(split, labels, feature, direction):
    if direction == 'left':
        symbol = '<='
    else:
        symbol = '>'
    if len(labels) == 1:
        if list(labels.keys())[0] == b'negative':
            print(feature, symbol, format(split, '.6f'), '[' + str(
                labels[b'negative']) + ' ' + '0' + ']')
        else:
            print(
                feature, symbol, format(split, '.6f'), '[' + '0' + ' ' + str(
                    labels[b'positive']) + ']')
    else:
        print(feature, symbol, format(split, '.6f'), '[' + str(
            labels[b'negative']) + ' ' +
              str(labels[b'positive']) + ']')


def PrintNumericLeaf(root, split, labels, feature, direction):
    if direction == 'left':
        symbol = '<='
    else:
        symbol = '>'
    if len(labels) == 1:
        if list(labels.keys())[0] == b'negative':
            print(feature, symbol, format(split, '.6f'), '[' + str(
                labels[b'negative']) + ' ' + '0' + ']:' + ' ' + 'negative')
        else:
            print(feature, symbol, format(split, '.6f'), '[' + '0' + ' ' + str(
                labels[b'positive']) + ']:' + ' ' + 'positive')
    else:
        print(feature, symbol, format(split, '.6f'), '[' + str(
            labels[b'negative']) + ' ' +
              str(labels[b'positive']) + ']', end='')
        print(':', DetermineClass(root))


def PreorderTree(root, level, direction):
    if root.children:
        print('|    ' * level, end='')
        split = root.parent.split
        labels = root.label
        feature = root.parent.feature
        if direction == 'left':
            PrintNumericNodes(split, labels, feature, direction)

        else:
            PrintNumericNodes(split, labels, feature, direction)
        if type(root.split) == float:
            PreorderTree(root.children[0], level + 1, direction='left')
            PreorderTree(root.children[1], level + 1, direction='right')
        else:
            for child in root.children:
                PrintNominalNodes(child, level + 1)
    else:
        labels = root.label
        feature = root.parent.feature
        split = root.parent.split
        print('|    ' * level, end='')
        if direction == 'left':
            PrintNumericLeaf(root, split, labels, feature, direction)
        else:
            PrintNumericLeaf(root, split, labels, feature, direction)


def PrintNominalNodes(root, level):
    print('|    ' * level, end='')
    split = root.parent.split
    labels = root.label
    feature = root.parent.feature
    Index = root.parent.children.index(root)
    if root.children:
        if len(labels) == 1:
            if list(labels.keys())[0] == b'negative':
                print(feature, '=', split[Index], '[' + str(
                    labels[b'negative']) + ' ' + '0' + ']')
            else:
                print(feature, '=', split[Index], '[' + '0' + ' ' + str(
                    labels[b'positive']) + ']')
        else:
            print(feature, '=', split[Index], '[' + str(
                labels[b'negative']) + ' ' + str(labels[b'positive']) + ']')
            printotherTree(root, level + 1)

    else:
        if len(labels) == 1:
            if list(labels.keys())[0] == b'negative':
                print(feature, '=', split[Index], '[' + str(
                    labels[b'negative']) + ' ' + '0' + ']:' + ' ' + 'negative')
            else:
                print(feature, '=', split[Index], '[' + '0' + ' ' + str(
                    labels[b'positive']) + ']:' + ' ' + 'positive')
        else:
            print(feature, '=', split[Index], '[' + str(
                labels[b'negative']) + ' ' + str(labels[b'positive']) + ']:', end='')
            print(DetermineClass(root))


def printotherTree(root, level):
    if type(root.split) == float:
        PreorderTree(root.children[0], level, direction='left')
        PreorderTree(root.children[1], level, direction='right')
    else:
        for child in root.children:
            PrintNominalNodes(child, level)


def PrintTree(root, level):
    if not root.parent:

        if type(root.split) == float:
            PreorderTree(root.children[0], level, direction='left')
            PreorderTree(root.children[1], level, direction='right')
        else:
            for child in root.children:
                PrintNominalNodes(child, level)

    pass


def Diff(list1, list2):
    count = 0
    for row in np.array([list1, list2]).T:
        if row[0] == row[1]:
            pass
        else:
            count = count + 1
    return count


def TestTree(traindata, Features, TestData, m):
    root = Node(None)
    MakeSubtree(root, traindata, Features, m)
    PrintTree(root, 0)
    print('<Predictions for the Test Set Instances>')
    Predict, Actual = [], []
    for index, row in TestData.iterrows():
        predict = ClassPrediction(row, root)
        actual = row['class'].decode("utf-8")
        print(str(index + 1) + ':' + ' ' + 'Actual:', end=' ')
        print(actual, end=' ')
        print('Predicted:', end=' ')
        print(predict)
        Predict.append(predict)
        Actual.append(actual)
    lenA = len(Actual)
    corr = lenA - Diff(Predict, Actual)
    print('Number of correctly classified:', end=' ')
    print(corr, end=' ')
    print('Total number of test instances:', end=' ')
    print(lenA)
    return corr, lenA

~~~
### Main Function
~~~
def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    m = int(sys.argv[3])
    TrainData = pd.DataFrame(arff.loadarff(train_file)[0])
    Features = arff.loadarff(train_file)[1]
    TestData = pd.DataFrame(arff.loadarff(test_file)[0])
    corr, lenA = TestTree(TrainData, Features, TestData, m)
~~~
