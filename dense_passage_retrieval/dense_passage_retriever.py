from .dpr_reader import DPRReader
from .dpr_index import DPRIndex


class DensePassageRetriever(DPRIndex, DPRReader):

    def __init__(self, documents):
        DPRIndex.__init__(self, documents)

    def read_dual_results(self, question:str, dual_results: list):
        '''Augments the results from search_dual_index() with the DPR reader.'''
        chunks = [result['chunk'] for result in dual_results]
        titles = [result['document']['title'] for result in dual_results]
        reader_results = self.read_documents(question, chunks, titles)
        for dual_result, reader_result in zip(dual_results, reader_results):
            dual_result['answer'] = reader_result['answer']
            dual_result['scores']['reader_relevance'] = reader_result['relevance']
            dual_result['scores']['answer_score'] = reader_result['answer_score']
        dual_results.sort(key=lambda x: -x['scores']['reader_relevance'])
        return dual_results

    def search(self, question:str):
        '''One wrapper to rule them all.'''
        #TODO: Group on document level.
        dual_results = self.search_dual_index(question)
        full_results = self.read_dual_results(question, dual_results)
        return full_results
