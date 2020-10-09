import logging
from typing import List
from tqdm import tqdm
from elasticsearch import Elasticsearch

class ElasticIndex:

    INDEX_NAME = 'dense-passage-retrieval'

    def __init__(self, documents: List[str]):
        self.es = Elasticsearch()
        if self.es.indices.exists(self.INDEX_NAME):
            logging.warning(f'Deleting old index for {self.INDEX_NAME}.')
            self.es.indices.delete(self.INDEX_NAME)
        self.es.indices.create(index=self.INDEX_NAME)
        for i, document in tqdm(enumerate(documents), total=len(documents)):
            document = {
                'document': document
            }
            self.es.create(self.INDEX_NAME, id=i, body=document)

    def search_index(self, query):
        body = {
            'size': 10,
            'query': {
                'match': {
                    'document': query
                }
            }
        }
        results = self.es.search(index=self.INDEX_NAME, body=body)
        hits = results['hits']['hits']
        return hits
