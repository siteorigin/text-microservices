# Text Encoder


## Setup

Install python modules and nltk data for tokenizer:

```shell
pip install -r requirements.txt
python -m nltk.downloader 'punkt'
./text-microservices/install_models.sh
sudo apt install gunicorn
```

Startup script

```shell
pip install -r /path/to/text-microservices/requirements.txt
python -m nltk.downloader 'punkt'
sudo -b gunicorn -b :80 --chdir /path/to/text-microservices/src main:app
```

## API Documentation

The API support both `GET` and `POST`, the arguments are:

1. `text`: the text to be encoded
2. `type`: only `text` is supported for now
3. `model`: the encoder model, only `cbow-glove` is supported for the moment

Return data will in json format.
```
// In the case of login fail
{
	'status' : -1
}

// In the case of error in encoder
{
	'status' : 1
}

// In the case of success
{
	'status' : 0,
	'features' : [ /* a feature vector for the entire document */ ],
	'sentences' : [
		{
			'status' : 0,
			'text' : 'This is the first sentence.',
			'features' : [ /* a feature vector for the sentence */ ],
			'salience' : 0.5,
		},
		{
			'status' : 0,
			'text' : 'This is the second sentence.',
			'features' : [ /* a feature vector for the sentence */ ],
			'salience' : 0.85,
		},
		{
			'status' : 1,
			'text' : 'This-is the-third sentence-that all-words-are-unknown',
		},
		// The rest of the sentences, in the order they appeard in the document
	]
}
```
