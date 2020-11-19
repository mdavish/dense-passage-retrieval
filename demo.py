import streamlit as st
import requests

'''
# Dense Passage Retrieval
'''

max_chunk_length_chars = 250

@st.cache()
def query_server(query):
    endpoint = 'http://127.0.0.1:5000/search'
    params = {
        'query': query
    }
    results = requests.get(endpoint, params)
    return results.json()

answers_score_threshold = st.sidebar.slider('Answer Score Threshold',
                                            0.0, -10.0, 5.0)
reader_relevance_threshold = st.sidebar.slider('Reader Relevance Threshold',
                                               0.0, -10.0, 5.0)


def result_card(i, result):
    st.write(f"### {str(i)}. {result['document']['title']}")
    answer = result['answer']
    good_score = result['scores']['answer_score'] > answers_score_threshold
    if good_score and answer:
        st.write(f"**Answer:** {result['answer']}")
    else:
        st.write('_No Answer Detected_')
    if len(result['chunk']) > max_chunk_length_chars:
        st.write(f"{result['chunk'][:max_chunk_length_chars]}...")
    else:
        st.write(f"{result['chunk']}")
    st.json(result['scores'])
    with st.beta_expander('See full document'):
        st.write(result['document']['body'])

query = st.text_input('Query')
if query:
    results = query_server(query)
    '# Results'
    for i, result in enumerate(results, 1):
        result_card(i, result)
    '## Full JSON'
    st.json(results)
