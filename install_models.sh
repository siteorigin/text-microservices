#!/bin/bash

mkdir -p ./models/
mkdir -p ./models/cbow

# Do the character word2vec model
wget https://storage.googleapis.com/text-microservice-models/char_word2vec.tar.gz -P ./models
tar -xvf ./models/char_word2vec.tar.gz -C ./models/
rm ./models/char_word2vec.tar.gz

# Setup the CBOW model
wget https://storage.googleapis.com/text-microservice-models/glove.840B.300d.zip -P ./models/cbow/
unzip ./models/cbow/glove.840B.300d.zip -d ./models/cbow/
awk '{print $1}' ./models/cbow/glove.840B.300d.txt > ./models/cbow/glove.840B.300d.vocab.txt
rm ./models/cbow/glove.840B.300d.zip

# Setup the skip thought model
wget --quiet https://storage.googleapis.com/text-microservice-models/skip_thoughts_uni_2017_02_02.tar.gz -P ./models
tar -xvf ./models/skip_thoughts_uni_2017_02_02.tar.gz -C ./models/
python ./process_src/process_skip-thought.py
rm ./models/skip_thoughts_uni_2017_02_02.tar.gz