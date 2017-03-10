# Text Encoder


## Setup

Install python modules and nltk data for tokenizer:

```shell
pip install -r requirements.txt
python -m nltk.downloader 'punkt'
```

Download Glove and some preprocess:

```shell
>>> mkdir -p models/cbow/
>>> wget http://nlp.stanford.edu/data/glove.840B.300d.zip -P models/cbow/
>>> cd models/cbow/ && unzip glove.840B.300d.zip```
>>> awk '{print $1}' glove.840B.300d.txt > glove.840B.300d.vocab.txt
```

Start Tornado server, simply cd into `src` and run:
```
python main.py
```
