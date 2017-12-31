# Perceptron Text Classification
A perceptron based text classification using word bag as feature extractions and applied on a sentiment analysis dataset.

## How to use ?
### Prepare your dataset: A txt file including labeled sentences as follow:
1 this is positive text
<br>1 this another positive text
<br>0 this a negative text
<br>0 this another negative text
<br>1 etc..
<br>0 etc..
### Launch the script train_and_test.py
python train_and_test.py -t ../sentiement_analysis.txt  -i 10 -r 0.1 -s 10000
<br>To get help:python
<br>train_and_test.py -h
<br>usage: getTheBag_3.py [-h] -t TEXT_FILE -i NUMBER_OF_ITERATIONS -r<br>
                      LEARNING_RATE -s SHUFFLE_FACTOR

optional arguments:<br>
  -h, --help            show this help message and exit
  -t TEXT_FILE, --text_file TEXT_FILE
                        Labled text file, line example:"1 This is a positive
                        sentence"
  -i NUMBER_OF_ITERATIONS, --number_of_iterations NUMBER_OF_ITERATIONS
                        Number of training iterations
  -r LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate
  -s SHUFFLE_FACTOR, --shuffle_factor SHUFFLE_FACTOR
                        Date shuffling factor
### Datasets
A very interesting dataset to test this classifier could be found in Kaggle: https://www.kaggle.com/c/si650winter11/data


