# Author: walid.daboubi@gmail.com
# Version: 1.0 - 2017/12/24
# About: perceptron algorithm applied on sentiment analysis

from utilities import *
from args import *

ITERATIONS = int(args.number_of_iterations)
LEARNING_RATE = float(args.learning_rate)
SHUFFLE_COEF = int(args.shuffle_factor)
DATA_SPLIT_COEF = 0.9

# Characters to be removed from the raw text
TO_BE_CLEANED = ['*','<','>','\'','\"','[',']','\0','\1','\2','\3','\4','\5','\6','\7','\8','\9', '.', '/', ':', ';','!','?','(',')','&','%','+','-']

# Read data file
data_file = args.text_file

# Read data from file
raw_data = open(data_file, 'r')

# Clean the raw text line by line
cleaned_text = clean_text(raw_data, TO_BE_CLEANED)

# Shuffle data
ready_data = shuffle_data(cleaned_text, SHUFFLE_COEF)

# Get the words bag and training data
word_bag, X, Y,trainig_phrases = get_data(ready_data, 0,int(len(ready_data)*DATA_SPLIT_COEF), None)

# Number of training examples

m = X.shape[1]

# Train
w, b = train_model(m, ITERATIONS, LEARNING_RATE, Y, X)

# Get the testing data
testing_X, testing_Y,phrases = get_data(ready_data, int(len(ready_data)*DATA_SPLIT_COEF)+1, len(ready_data)-1, word_bag)

# Test
A = sigmoid(np.dot(testing_X, w) + b)

# Statistics
TP, FP, TN, FN, accuracy, recall, precision, f1 = get_precision_stats(A, testing_Y)


print "------------------------------------------------------------------------------------------"
print "Data inputs number: " + str(len(testing_Y))
print "True positives: " + str(TP)
print "True negatives: " + str(TN)
print "False positives: " + str(FP)
print "False negatives: " + str(FN)
print "Accuracy: " + str(accuracy)
print "Recall: " + str(recall)
print "Precision: " + str(precision)
print "F1: " + str(f1)

#for f in phrases:
#    print f
