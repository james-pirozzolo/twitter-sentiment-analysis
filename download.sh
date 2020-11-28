#!/bin/bash

# this downloads the zip file that contains the data
curl http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip -L -o data.zip
# his unzips the zip file - you will get a directory named "data" containing the data
unzip data.zip
# this cleans up the zip file, as we will no longer use it
rm data.zip
# rename files to test.csv and train.csv and move to data folder
mkdir data
mv testdata.manual.2009.06.14.csv data/test.csv
mv training.1600000.processed.noemoticon.csv data/train.csv

echo downloaded data