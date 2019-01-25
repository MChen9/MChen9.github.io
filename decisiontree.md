---
layout: page
title: Fundamental Algorithms
subtitle: Decision Tree
---
~~~
import pandas as pd
import numpy as np
from scipy.io import arff
import sys
~~~

### Tree Construction
~~~
class Tree(object):
    def __init__(self, value=None, sub=[], type=None, level=None, sign=None, typecounts=None, info=None, cls=None,
                 parent=None):  # type is the selected feature

        self.sub = sub  # sub tree
        self.value = value  # node feature value(not feature but the partition value), can be used to classify testing samples for numeric is the threshold
        self.type = type  # type of feature to split A1,A2...etc
        self.level = level  # level of the node
        self.sign = sign  # less or greater for numeric splits
        self.typecounts = typecounts  # count of instances of each class
        self.info = info  # cross entropy of the node
        self.stopcrt = None
        self.cls = cls
        self.parent = parent
~~~

### Convert Bytes Type into String
~~~
def convert(x):
    return str(x, encoding="utf-8")
~~~

### Calculate Information Gain
~~~

def info(num):
    # list of 2 [a,b]
    # #return entropy of the candidate subtree
    num = list(num)
    if len(num) == 1:
        return 0
    if not num:
        return np.inf
    else:

        a = num[0]
        b = num[1]
        if a == 0 and b == 0:
            return None
        if not a * b:
            return 0
        s = sum(num)
        inf = -a / s * np.log2(a / s) - b / s * np.log2(b / s)
        return inf

~~~

### for Numerical Feature
~~~
def find_numeric_threshold(cutdata, feature):
    # return threshold of the selected feature
    numeric_value = sorted(cutdata[feature].values)  # return a unique sorted array
    threshold = [0] * (len(numeric_value) - 1)
    infomin = np.inf
    # if cutdata.shape[0]==15:
    #    print('breakpoint')
    for i in range(1, len(numeric_value)):
        threshold[i - 1] = (numeric_value[i - 1] + numeric_value[i]) / 2.0
    # determin which is the best threshold.
    if not threshold:
        thresholdSelected = None
        return thresholdSelected, infomin
    threshold = sorted(list(set(threshold)))
    thresholdSelected = threshold[0]
    totalLength = cutdata[feature].shape[0]
    for j in threshold:
        upPart = cutdata[cutdata[feature] > j]['class'].value_counts()
        loPart = cutdata[cutdata[feature] <= j]['class'].value_counts()
        ptinf = info(upPart) * sum(upPart) / totalLength + info(loPart) * sum(loPart) / totalLength
        if infomin > ptinf:
            infomin = ptinf
            thresholdSelected = j
    return thresholdSelected, infomin

~~~
### Algorithm Implementation
~~~
def classify(root, feature, cutdata, feature_type):  # the current root, features not used, current data, featuretype.
    infomin = np.inf
    numerica_threshold = {}
    instanceNumberAfterP = {}
    for i in feature:
        ptinf = 0

        if 1:  # not i == root.value:  # root feature
            tempSubInstNumber = {}
            if feature_type[i][0] == 'nominal':
                partitions = list(cutdata.groupby([i]).groups.keys())

                for j in partitions:
                    clss = class_count(cutdata[df[i] == j], feature_type['class'][1])
                    tempSubInstNumber[j] = clss
                    ptinf += info(clss) * (sum(clss) / cutdata[i].shape[0])
                    if np.isnan(ptinf):
                        pass
                    # '+'in df[df['class']=='-']['class'].value_counts()
            elif feature_type[i][0] == 'numeric':
                partitions, ptinf = find_numeric_threshold(cutdata, i)
                numerica_threshold[i] = partitions
                if np.isnan(ptinf):
                    pass
                for j in ['g', 'l']:
                    if j == 'g':
                        clss_data = cutdata[cutdata[i] > partitions]
                    elif j == 'l':
                        clss_data = cutdata[cutdata[i] <= partitions]
                    tempSubInstNumber[j] = class_count(clss_data, feature_type['class'][1])
                    # gain = 0  # this gain is info after classfication, the smaller the better

            else:
                # print('Unknown feature type, exit')
                # exit()
                pass
            instanceNumberAfterP[i] = tempSubInstNumber
        if infomin > ptinf:
            infomin = ptinf
            selected = i
            instanceNumberAfterP[i] = tempSubInstNumber
    return infomin, selected, numerica_threshold, instanceNumberAfterP[selected]
~~~
### Stop Critiron
~~~

def STOP(cutdata, root, level, m, feature_type):
    stop = False
    if not root.typecounts[0] * root.typecounts[1]:  # 1
        root.stopcrt = 1
        stop = True
    elif sum(root.typecounts) < m:  # 2
        root.stopcrt = 2
        stop = True

    elif len(cutdata.columns) == 2:  # 4
        if feature_type[0] == 'nominal':
            stop = True  # if there is only one nominal feature left, we don't have more splits
        root.stopcrt = 4
    else:
        pass
    return stop

~~~
~~~
def det_class(root, class_list):  # class name list
    if not root.typecounts[0] == root.typecounts[1]:
        root.cls = class_list[np.argmax(root.typecounts)]
    else:
        det_class(root.parent, class_list)
        root.cls = root.parent.cls

~~~

~~~
def Make_sub_tree(cutdata, root, feature_type, level):
    # ifStop()
    class_list = feature_type['class'][1]
    if STOP(cutdata, root, level, m, feature_type):
        det_class(root, class_list)
        return
    # break
    else:
        infoTreeSplit, selectedFeature, numerica_threshold, classnumbers = classify(root, cutdata.columns[
                                                                                          0:len(cutdata.columns) - 1],
                                                                                    cutdata, feature_type)
        # print(infoTreeSplit, selectedFeature, numerica_threshold)

        if root.info <= infoTreeSplit:  # 3
            root.stopcrt = 3
            det_class(root, class_list)
            return
        if feature_type[selectedFeature][0] == 'nominal':
            for featureEnumerate in feature_type[selectedFeature][1]:
                existlist = list(cutdata.groupby(selectedFeature).groups.keys())
                if featureEnumerate in existlist:
                    root.sub.append(Tree(value=featureEnumerate, sub=[], type=selectedFeature, level=level + 1,
                                         typecounts=classnumbers[featureEnumerate],
                                         info=info(classnumbers[featureEnumerate]), parent=root))
                    Make_sub_tree(cutdata[cutdata[selectedFeature] == featureEnumerate].drop(selectedFeature, axis=1),
                                  root.sub[-1], feature_type, level + 1)
                else:  # deal with the feature with empty instances
                    root.sub.append(Tree(value=featureEnumerate, sub=[], type=selectedFeature, level=level + 1,
                                         typecounts=[0, 0],
                                         info=info([0, 0]), parent=root))
                    root.sub[-1].stopcrt = 1
                    det_class(root.sub[-1], class_list)
                    return  # 1
                    # df[df['A1'] == 'b'].drop('A1', axis=1)
        elif feature_type[selectedFeature][0] == 'numeric':
            root.sub.append(
                Tree(value=numerica_threshold[selectedFeature], sub=[], type=selectedFeature, level=level + 1,
                     sign='l', typecounts=classnumbers['l'], info=info(classnumbers['l']), parent=root))
            Make_sub_tree(
                cutdata[cutdata[selectedFeature] <= numerica_threshold[selectedFeature]],
                root.sub[-1], feature_type, level + 1)
            root.sub.append(
                Tree(value=numerica_threshold[selectedFeature], sub=[], type=selectedFeature, level=level + 1,
                     sign='g', typecounts=classnumbers['g'], info=info(classnumbers['g']),
                     parent=root))  # l for less or equal , g for greater
            Make_sub_tree(
                cutdata[cutdata[selectedFeature] > numerica_threshold[selectedFeature]],
                root.sub[-1], feature_type, level + 1)
~~~
### Print
~~~

def print_tree(root, feature_type):  # loop
    if root.level > 0:
        print('|	' * (root.level - 1), end='')
        print_single_tree(root, feature_type)
    if root.sub:
        for j in root.sub:
            print_tree(j, feature_type)


def print_single_tree(root, feature_type):
    if not root.sub:
        if feature_type[root.type][0] == 'numeric':
            if root.sign == 'l':
                print(root.type.lower(), '<=', "{0:.6f}".format(root.value),
                      '[' + str(root.typecounts[0]) + ' ' + str(root.typecounts[1]) + ']', end='')
                print(':', root.cls)
            elif root.sign == 'g':
                print(root.type.lower(), '>', "{0:.6f}".format(root.value),
                      '[' + str(root.typecounts[0]) + ' ' + str(root.typecounts[1]) + ']', end='')
                print(':', root.cls)
            else:
                print('ERROR')
        elif feature_type[root.type][0] == 'nominal':
            print(root.type.lower(), '=', root.value,
                  '[' + str(root.typecounts[0]) + ' ' + str(root.typecounts[1]) + ']', end='')
            print(':', root.cls)
        else:
            print('ERROR')
    else:  # if leaf, we don't print the class out.
        if feature_type[root.type][0] == 'numeric':
            if root.sign == 'l':
                print(root.type.lower(), '<=', "{0:.6f}".format(root.value),
                      '[' + str(root.typecounts[0]) + ' ' + str(root.typecounts[1]) + ']')
            elif root.sign == 'g':
                print(root.type.lower(), '>', "{0:.6f}".format(root.value),
                      '[' + str(root.typecounts[0]) + ' ' + str(root.typecounts[1]) + ']')
            else:
                print('ERROR')
        elif feature_type[root.type][0] == 'nominal':
            print(root.type.lower(), '=', root.value,
                  '[' + str(root.typecounts[0]) + ' ' + str(root.typecounts[1]) + ']')
        else:
            print('ERROR')

~~~
### Prediction
~~~
def predict_class(root, testData, feature_type):
    if root.sub:
        FeatureSplit = root.sub[0].type
        a = testData[FeatureSplit]
        if feature_type[FeatureSplit][0] == 'nominal':
            for sub in root.sub:
                if a.values[0] == sub.value:
                    return predict_class(sub, testData, feature_type)
            pass
        else:
            if not len(a) == 1:
                print('ERROR in a length')
            if a.item() <= root.sub[0].value:
                return predict_class(root.sub[0], testData, feature_type)
            else:
                return predict_class(root.sub[1], testData, feature_type)
    else:
        return root.cls

    pass


def class_count(data, class_list):  # string of class attribute
    cls_count = [0] * 2
    raw_list = data['class'].value_counts()
    if class_list[0] in raw_list:
        cls_count[0] = int(raw_list[class_list[0]])
    if class_list[1] in raw_list:
        cls_count[1] = int(raw_list[class_list[1]])
    return cls_count  # list of length 2, order: first class in class lsit, second class in class list.
~~~

### Main
~~~

if __name__ == '__main__':

    m = int(sys.argv[3])
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    '''
    m = 10
    train_file = 'credit_train.arff'
    test_file = 'credit_test.arff'
    '''
    data = arff.loadarff(train_file)

    df = pd.DataFrame(data[0])  # .drop('A3',axis=1)
    feature_type = data[1]
    for i in df.columns:
        if feature_type[i][0] == 'nominal':
            df[i] = df[i].apply(convert)
    class_list = data[1]['class'][1]
    rootInfo = class_count(df, class_list)

    # -----------------------generate root------------------
    Root = Tree(level=0, typecounts=rootInfo, info=info(rootInfo))  # this is the constructed tree
    Make_sub_tree(df, Root, data[1], 0)
    # --------------print out the constructed tree
    print_tree(Root, data[1])
    # ---------------------prediction.
    # convert the test data into df
    datatest = arff.loadarff(test_file)
    dft = pd.DataFrame(datatest[0])  # .drop('A3',axis=1)
    feature_typetest = datatest[1]
    for i in dft.columns:
        if feature_typetest[i][0] == 'nominal':
            dft[i] = dft[i].apply(convert)
    correctPrediction = 0
    print('<Predictions for the Test Set Instances>')
    for i in range(dft.shape[0]):
        instance = dft.iloc[[i]]  #
        print(str(i + 1) + ':', 'Actual:', instance['class'].values[0], end=' ')
        predicted = predict_class(Root, instance.drop('class', axis=1), feature_typetest)
        print('Predicted:', predicted)
        if predicted == instance['class'].values[0]:
            correctPrediction += 1
        pass
    print('Number of correctly classified:', correctPrediction, 'Total number of test instances:', dft.shape[0], end='')
    pass
~~~
