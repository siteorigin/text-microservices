# Use a version of Tensorflow that works with our system.
FROM tensorflow/tensorflow:1.0.1

COPY . .

# Setup all the required Python extras
RUN pip install -r ./requirements.txt && \
	python -m nltk.downloader 'punkt'

ADD https://storage.googleapis.com/text-microservice-models/text-microservice-models.tar.gz ./text-microservice-models.tar.gz

# Extract the models
RUN tar -xvf ./text-microservice-models.tar.gz && \
	rm ./text-microservice-models.tar.gz

# Run the Gunicorn App
ENTRYPOINT gunicorn -b :8080 --chdir ./src main:app