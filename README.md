# A single perceptron image classification from scratch
## About
A perceptron based text classification using bag of words as features extractor and applied on a sentiment analysis dataset.
This article http://enigmater.blogspot.com/2018/01/perceptron-based-text-classifier-from.html explains how it works.
## How to use ?
### Prepare your dataset: A txt file including labeled sentences as follow:
1 this is positive text
<br>1    this another positive text
<br>0    this a negative text
<br>0    this another negative text
<br>1    etc..
<br>0    etc..
### Launch the script train_and_test.py
#### Example
```
$ python train_and_test.py -t ../sentiement_analysis.txt  -i 10 -r 0.1 -s 10000
```
#### To get help
```
$ python train_and_test.py -h
```
### Datasets
A very interesting dataset to test this classifier can be found at https://www.kaggle.com/c/si650winter11/data


