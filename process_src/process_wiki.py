#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import codecs
from collections import defaultdict
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from gensim.corpora.wikicorpus import *

def tokenize(content):
    # override original method in wikicorpus.py
    sents = sent_tokenize(content)
    sents = [word_tokenize(sent.strip()) for sent in sents]
    return sents

def process_article(args):
   # override original method in wikicorpus.py
    text, lemmatize, title, pageid = args
    text = filter_wiki(text)
    if lemmatize:
        result = utils.lemmatize(text)
    else:
        result = tokenize(text)
    return result, title, pageid


class MyWikiCorpus(WikiCorpus):
    def __init__(self, fname, processes=None, lemmatize=False, dictionary=None, filter_namespaces=('0',)):
        WikiCorpus.__init__(self, fname, processes, lemmatize, dictionary, filter_namespaces)

    def get_texts(self):
        articles, articles_all = 0, 0
        positions, positions_all = 0, 0
        texts = ((text, self.lemmatize, title, pageid) for title, text, pageid in extract_pages(bz2.BZ2File(self.fname), self.filter_namespaces))
        pool = multiprocessing.Pool(self.processes)
        # process the corpus in smaller chunks of docs, because multiprocessing.Pool
        # is dumb and would load the entire input into RAM at once...
        for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
            for tokens, title, pageid in pool.imap(process_article, group):  # chunksize=10):
                articles_all += 1
                positions_all += len(tokens)
                # article redirects and short stubs are pruned here
                '''
                if len(tokens) < ARTICLE_MIN_WORDS or any(title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):
                    continue
                '''
                if len(tokens) < ARTICLE_MIN_WORDS:
                    continue
                articles += 1
                positions += len(tokens)
                if self.metadata:
                    yield (tokens, (pageid, title))
                else:
                    yield tokens
        pool.terminate()

        logger.info(
            "finished iterating over Wikipedia corpus of %i documents with %i positions"
            " (total %i articles, %i positions before pruning articles shorter than %i words)",
            articles, positions, articles_all, positions_all, ARTICLE_MIN_WORDS)
        self.length = articles  # cache corpus length

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print 'Usage: python process_wiki.py in_file out_file'
        sys.exit(1)
    inp, outp = sys.argv[1:3]

    max_document = 2000
    n_words = 100000
    word_cnt = defaultdict(int)
    wiki = MyWikiCorpus(inp, lemmatize=False, dictionary={})

    i = 0
    for text in wiki.get_texts():
        for sent in text:
            for w in sent:
                word_cnt[w] += 1
        i = i + 1
        if (i % 1000 == 0):
            logger.info("Processed " + str(i) + " articles")
        if (i > max_document):
            break
    words = [w for w in sorted(word_cnt, key=word_cnt.get, reverse=True)]
    words = set(words[:n_words])

    i = 0
    output = codecs.open(outp, 'w', 'utf-8')
    for text in wiki.get_texts():
        new_text = []
        for sent in text:
            sent = [w if w in words else '<unk>' for w in sent]
            new_text.append(' '.join(sent))
        output.write('\n'.join(new_text) + '\n\n')
        i = i + 1
        if (i % 1000 == 0):
            logger.info("Saved " + str(i) + " articles")
        if (i > max_document):
            break

    output.close()
    logger.info("Finished Saved " + str(i) + " articles")
