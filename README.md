# Text Encoder


## Setup

Install python modules and nltk data for tokenizer:

```shell
>>> pip install -r requirements.txt
>>> python -m nltk.downloader 'punkt'
```

Model files for `cbow-glove`, download Glove and preprocessing:

```shell
>>> mkdir -p models/cbow/
>>> wget http://nlp.stanford.edu/data/glove.840B.300d.zip -P models/cbow/
>>> cd models/cbow/ && unzip glove.840B.300d.zip
>>> awk '{print $1}' glove.840B.300d.txt > glove.840B.300d.vocab.txt
```

Model files for `skip-thought`, simply put the trained model files into `models/skip-thought`

Start Flask server, simply cd into `src` and run, the API will listen on `127.0.0.1:5000`
```
python main.py
```

To deploy with authorization, simply set environment variable `SONAR_AUTH_REQUESTS`.

To deploy on Google App Engine, use `src/app.yaml`.

## Document

The API support both `GET`(url parameters) and `POST`(request in json format), the arguments are:

1. `text`: the text to be encoded
2. `type`: only `text` is supported for now
3. `model`: the encoder model, only `cbow-glove` is supported for the moment

If environment variable `SONAR_AUTH_REQUESTS` is set, you should provide these arguments for authorization:

1. `user_email`: The user's email address. For security this should always be sent as an MD5 encoded string.
2. `key`: a key return by SiteOrigin.com 
3. `key_expire`: an expiry timestamp for the key

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
