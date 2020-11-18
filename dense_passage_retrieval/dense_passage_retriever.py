from .dpr_reader import DPRReader
from .dpr_index import DPRIndex


class DensePassageRetriever(DPRIndex, DPRReader):

    def __init__(self, documents):
        DPRIndex.__init__(self, documents)

    def read_dual_results(self, dual_results: dict):
        '''Augments the results from search_dual_index() with the DPR reader.'''
        pass
