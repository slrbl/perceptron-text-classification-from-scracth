# Author: walid.daboubi@gmail.com
# Version: 1.0 - 2017/12/24
# About: perceptron algorithm applied on sentiment analysis

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t',
                    '--text_file',
                    help='Labled text file, line example:"1 This is a positive sentence" ',
                    required=True
                    ),
parser.add_argument('-i',
                    '--number_of_iterations',
                    help='Number of training iterations',
                    required=True
                    )
parser.add_argument('-r',
                    '--learning_rate',
                    help='Learning rate',
                    required=True
                    )
parser.add_argument('-s',
                    '--shuffle_factor',
                    help='Date shuffling factor',
                    required=True
                    )

args = parser.parse_args()
