# Use a version of Tensorflow that works with our system.
FROM tensorflow/tensorflow:1.0.1

COPY . .

# Setup all the required Python extras
RUN pip install -r ./requirements.txt && \
	python -m nltk.downloader 'punkt'


RUN apt-get update && \
	apt-get -f install && \
	apt-get install wget

# Create all the folders and download the models
RUN mkdir -p ./models/ && \
	mkdir -p ./models/cbow && \
	wget --quiet https://storage.googleapis.com/text-microservice-models/char_word2vec.tar.gz -P ./models && \
	wget --quiet https://storage.googleapis.com/text-microservice-models/glove.840B.300d.zip -P ./models/cbow/ && \
	wget --quiet https://storage.googleapis.com/text-microservice-models/skip_thoughts_uni_2017_02_02.tar.gz -P ./models

# Setup the character word2vec model
RUN tar -xvf ./models/char_word2vec.tar.gz -C ./models/ && \
	rm ./models/char_word2vec.tar.gz
	
# Setup the CBOW model
RUN unzip ./models/cbow/glove.840B.300d.zip -d ./models/cbow/ && \
	awk '{print $1}' ./models/cbow/glove.840B.300d.txt > ./models/cbow/glove.840B.300d.vocab.txt && \
	rm ./models/cbow/glove.840B.300d.zip

# Setup the skip thought model
RUN tar -xvf ./models/skip_thoughts_uni_2017_02_02.tar.gz -C ./models/ && \
	python ./process_src/process_skip-thought.py && \
	rm ./models/skip_thoughts_uni_2017_02_02.tar.gz

# Run the Gunicorn App
ENTRYPOINT gunicorn -b :8080 --chdir ./src main:app
