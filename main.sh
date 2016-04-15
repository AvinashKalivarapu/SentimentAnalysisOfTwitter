#!/bin/bash

#python preprocess.py traintest.tsv preprocessedTweets.txt
echo "Preprocessing successfull !! (Saved to preprocessedTweets.txt)"
#python preprocess.py testing.tsv testpreprocessedTweets.txt
echo "Preprocessing successfull !! (Saved to testpreprocessedTweets.txt)"
python unigramSVM.py
