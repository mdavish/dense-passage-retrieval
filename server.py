import json
from flask import Flask, request, jsonify
from dense_passage_retrieval.dense_passage_retriever import DensePassageRetriever
from dense_passage_retrieval.dpr_document import DPRDocument

app = Flask(__name__)
TEST=False


def load_sample_data():
    with open('sample_data.json', 'r') as file:
        all_entities = json.load(file)
    blog_entities = [entity for entity in all_entities \
                     if entity['meta']['entityType'] == 'ce_blog']
    dpr_dicts = [{'title': blog['name'], 'body': blog['c_contents']}
                 for blog in blog_entities]
    dpr_docs = [DPRDocument(**dpr_dict) for dpr_dict in dpr_dicts]
    return dpr_docs

def setup_dpr():
    dpr_docs = load_sample_data()
    if TEST: dpr_docs = dpr_docs[:10]
    dpr = DensePassageRetriever(dpr_docs)
    return dpr

dpr = setup_dpr()
@app.route('/search')
def search():
    query = request.args.get('query')
    results = dpr.search(query)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
