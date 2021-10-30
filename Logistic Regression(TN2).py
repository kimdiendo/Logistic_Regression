from csv import reader
import numpy as np
import pandas as pd
import math


def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Load CSV file
filename = 'pima-indians-diabetes.data.csv'
dataset = load_csv(filename)
# Convert string to float
for i in range(len(dataset[0]) - 3):
    str_column_to_float(dataset, i)
# Alter each row by deleting ' ' quotes
data_update = dataset
for i in range(len(data_update)):
    data_update[i] = dataset[i][0:len(dataset[i]) - 3]
# Count amount of true and false values
dem = 0
dem1 = 0
for i in range(len(data_update)):
    if data_update[i][len(data_update[0]) - 1] == 1.0:
        dem = dem + 1
    else:
        dem1 = dem1 + 1
data_true = data_update[0:dem]
data_false = data_update[0:dem1]
j = 0
m = 0
# Create almost true values of data and almost false values of data
for i in range(len(data_update)):
    if data_update[i][len(data_update[0]) - 1] == 1.0:
        data_true[j] = data_update[i]
        j = j + 1
    else:
        data_false[m] = data_update[i]
        m = m + 1
data_train_true = data_true[0:200]
data_train_false = data_false[0:50]
data_test_true = data_true[200:250]
data_test_false = data_false[200:250]
data_update_train = data_train_true + data_train_false  # Include 200 true values and 50 false values for training
# model
data_update_test = data_test_true + data_test_false  # Include 50 true values and 50 false values for testing model
a = []  # Create array to contain maximum of each column
for i in range(len(data_update_train[0]) - 1):
    max = data_update_train[0][i]
    for j in range(len(data_update_train)):
        if max <= data_update_train[j][i]:
            max = data_update_train[j][i]
    a.append(max)
b1 = []  # Create array to contain minimum of each column
for i in range(len(data_update_train[0]) - 1):
    min = data_update_train[0][i]
    for j in range(len(data_update_train)):
        if min >= data_update_train[j][i]:
            min = data_update_train[j][i]
    b1.append(min)
# Min max normalization on training data
for i in range(len(data_update_train[0]) - 1):
    for j in range(len(data_update_train)):
        data_update_train[j][i] = float(data_update_train[j][i] - b1[i]) / (a[i] - b1[i])
for i in range(len(data_update_test[0]) - 1):
    for j in range(len(data_update_test)):
        data_update_test[j][i] = float(data_update_test[j][i] - b1[i]) / (a[i] - b1[i])
data_update_train = pd.read_csv("pima-indians-diabetes.data_update_train2&3.csv")
data_update_train = data_update_train.values
data_update_train = data_update_train.T
X_train = data_update_train[0:8]
Y_train = data_update_train[8]

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def model(X_train, Y_train, learning_rate):
    m = X_train.shape[1]  # Column of data
    n = X_train.shape[0]  # row of data
    W = np.random.random(n)  # Initialize Wi and B
    B = -1.0
    while True:
        Z = np.dot(W.T, X_train) + B  # Create new matrix with n*1
        A = Sigmoid(Z)
        Cost_Function = (1 / m) * np.sum(-Y_train * np.log(A) - (1 - Y_train) * np.log(1 - A))
        print("Value of cost after Optimization :" + str(Cost_Function))
        if Cost_Function < 0.53:
            break
        dW = (1 / m) * np.dot((A - Y_train), X_train.T)
        dB = (1 / m) * np.sum(A - Y_train)
        W = W - learning_rate * dW.T
        B = B - learning_rate * dB
    return W, B




learning_rate = 0.01
W, B = model(X_train, Y_train, learning_rate=learning_rate)
print("Configurations of Cost Function: ")
print(W)
print(B)
list_sigmod_test = []
for i in range(len(data_update_test)):
    y01 = 0.0
    for j in range(8):
        y01 = y01 + data_update_test[i][j] * W[j]
    list_sigmod_test.append(float(1 / (1 + math.pow(math.e, -(y01 + B)))))
for i in range(len(data_update_test)):
    if list_sigmod_test[i] >= 0.5:
        list_sigmod_test[i] = 1
        print(str(int(data_update_test[i][8])) + " : " + str(list_sigmod_test[i]))
    else:
        list_sigmod_test[i] = 0
        print(str(int(data_update_test[i][8])) + " : " + str(list_sigmod_test[i]))
true_positive = 0  # 1-1
true_negative = 0  # 0 -0
false_positive = 0  # 0-1
false_negative = 0  # 1-0
for i in range(len(data_update_test)):
    if int(data_update_test[i][8]) == 1 and list_sigmod_test[i] == 1:
        true_positive = true_positive + 1
    if int(data_update_test[i][8]) == 1 and list_sigmod_test[i] == 0:
        false_negative = false_negative + 1
    if int(data_update_test[i][8]) == 0 and list_sigmod_test[i] == 0:
        true_negative = true_negative + 1
    if int(data_update_test[i][8]) == 0 and list_sigmod_test[i] == 1:
        false_positive = false_positive + 1
Confusion_matrix = np.array([[true_positive, false_positive], [false_negative, true_negative]])
print("Confusion matrix :")
print(Confusion_matrix)
accuracy = (true_positive + true_negative) / (true_negative + true_positive + false_negative + false_positive)
print("Value of accuracy: " + str(int(accuracy * 100)) + "%")
precision = true_positive / (true_positive + false_positive)
print("Precision :" + str(precision))
recall = true_positive / (true_positive + false_negative)
print("Recall : " + str(recall))
F1 = (2 * precision * recall) / (precision + recall)
print("F1: " + str(F1))
