from dense_passage_retrieval.dense_passage_retriever import DensePassageRetriever
from dense_passage_retrieval.dpr_document import DPRDocument
import json
documents = [
    {
        'title': 'Andy Warhol',
        'body':
            'Andy Warhol was an American artist, film director, and '
            'producer who was a leading figure in the visual art movement known'
            ' as pop art.'

    },
    {
        'title': 'LeBron James',
        'body':
            'LeBron Raymone James Sr. is an American professional basketball '
            'player for the Los Angeles Lakers of the National Basketball '
            'Association. He is widely considered to be one of the greatest '
            'basketball players in NBA history.'

    },
    {
        'title': 'Jeff Bezos',
        'body':
            'Jeffrey Preston Bezos is an American internet entrepreneur, '
            'industrialist, media proprietor, and investor. He is best known as'
            ' the founder, CEO, and president of the multi-national technology '
            'company Amazon.'
    }
]
dpr_docs = [DPRDocument(**doc) for doc in documents]
dpr = DensePassageRetriever(dpr_docs)
results = dpr.search('who is a great athlete?')
print(json.dumps(results, indent=3))
