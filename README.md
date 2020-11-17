# Dense Passage Retrieval
This repository contains a user-friendly wrapper on top of HuggingFace's
[DPR](https://huggingface.co/transformers/model_doc/dpr.html) model, which is
based on Facebook AI's
[Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) paper.

This repo makes it a little simpler to work with these models and to use them
in conjunction with a BM25 or TFIDF ElasticSearch system, as is recommended
in the paper.

You can use the utilities like so:
```python
from dense_passage_retreival.dense_passage_retreiver import DensePassageRetriever
documents = [
    'Andy Warhol was an American artist, film director, and producer who was a '
    'leading figure in the visual art movement known as pop art.',
    'LeBron Raymone James Sr. is an American professional basketball player for'
    'the Los Angeles Lakers of the National Basketball Association. He is '
    'widely considered to be one of the greatest basketball players in NBA '
    'history.',
    'Jeffrey Preston Bezos is an American internet entrepreneur, industrialist,'
    'media proprietor, and investor. He is best known as the founder, CEO,'
    'and president of the multi-national technology company Amazon.'
]

dpr = DensePassageRetriever(documents)
dpr.search_dual_index('Who plays for the LA basketball team?')
```

In order to use this, you will need an Elastic Search cluster running.
You can find the instructions for this
[**here**](https://www.elastic.co/guide/en/elasticsearch/reference/current/targz.html).

Once installed, just navigate to your version and start it like this:
```
cd ~/elasticsearch-7.8.1/
./bin/elasticsearch
```
