# Perceptron Text Classification
A perceptron based text classification using word bag as feature extractions and applied on a sentiment analysis dataset.

### How to use ?
#### Prepare your dataset: A txt file including labeled sentences as follow:
1 this is positive text
<br>1 this another positive text
<br>0 this a negative text
<br>0 this another negative text
<br>1 etc..
<br>0 etc..
#### Launch the script train_and_test.py
Example: python train_and_test.py -t ../sentiement_analysis.txt  -i 10 -r 0.1 -s 10000


