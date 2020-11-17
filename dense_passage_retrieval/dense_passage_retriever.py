from .dpr_reader import DPRReader
from .dpr_index import DPRIndex


class DensePassageRetriever(DPRIndex, DPRReader):

    def __init__(self, documents):
        DPRIndex.__init__(self, documents)

    def _merge_results(self, sparse_result, dense_result):
        '''Merges the results of sparse and dense retrieval.'''
        dense_doc_ids = {res['document_id'] for res in dense_result}
        doc_results_dict = {}
        for doc_id in list(dense_doc_ids):
            doc_results_dict[doc_id] = {
                'returned_from_dense': True,
                'returned_from_sparse': False,
                'all_chunks': []
            }
            for result in dense_result:
                if result['document_id'] == doc_id:
                    chunk_record = {
                        'chunk': result['chunk'],
                        'chunk_id': result['chunk_id'],
                        'faiss_dist': result['faiss_dist']
                    }
                    doc_results_dict[doc_id]['all_chunks'].append(chunk_record)
            best_record = doc_results_dict[doc_id]['all_chunks'][0]
            doc_results_dict[doc_id]['best_chunk_id'] = best_record['chunk_id']
            doc_results_dict[doc_id]['best_chunk'] = best_record['chunk']
            doc_results_dict[doc_id]['best_faiss_distance'] = best_record['faiss_dist']
        for result in sparse_result:
            doc_id = int(result['_id'])
            if doc_id in doc_results_dict:
                doc_results_dict[doc_id]['returned_from_sparse'] = True
                doc_results_dict[doc_id]['elastic_score'] = result['_score']
            else:
                doc_results_dict[doc_id] = {
                    'returned_from_dense': False,
                    'returned_from_sparse': True,
                    'elastic_score': result['_score']
                }
        final_results = []
        for doc_id, value in doc_results_dict.items():
            result_dict = {
                **{'doc_id': doc_id, 'document': self.documents[doc_id]},
                **value
            }
            final_results.append(result_dict)
        return final_results

    def search_dual_index(self, query: str):
        '''Search both the sparse and dense indices and merge the results.'''
        sparse_result = self.search_sparse_index(query)
        dense_result = self.search_dense_index(query)
        merged_results = self._merge_results(sparse_result, dense_result)
        return merged_results

    def read_dual_results(self, dual_results: dict):
        '''Augments the results from search_dual_index() with the DPR reader.'''
        pass
