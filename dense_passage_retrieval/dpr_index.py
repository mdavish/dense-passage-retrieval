import re
import logging
import torch
import faiss
from tqdm import tqdm
from typing import List
from transformers import (DPRContextEncoder, DPRContextEncoderTokenizer,
                          DPRQuestionEncoder, DPRQuestionEncoderTokenizer)
from .document_chunker import DocumentChunker


class DPRIndex(DocumentChunker):

    '''
    Class for indexing and searching document and query vectors using the
    DPR model and FAISS for nearest neighbor.
    '''

    D = 768
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        'facebook/dpr-ctx_encoder-single-nq-base')
    context_model = DPRContextEncoder.from_pretrained(
        'facebook/dpr-ctx_encoder-single-nq-base', return_dict=True)
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        'facebook/dpr-question_encoder-single-nq-base')
    question_model = DPRQuestionEncoder.from_pretrained(
        'facebook/dpr-question_encoder-single-nq-base', return_dict=True)


    def __init__(self, documents: List[str]):
        super(DocumentChunker).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            self.reader_model = self.reader_model.cuda()
        self.faiss_index = faiss.IndexFlatL2(self.D)
        self.documents = documents
        self.chunks = []
        self.chunk_index = {}  # {chunk: document}
        self.inverse_chunk_index = {}  # {document: [chunks]}
        chunk_counter = 0
        for doc_counter, doc in tqdm(enumerate(documents),total=len(documents)):
            self.inverse_chunk_index[doc_counter] = []
            chunked_docs = self.chunk_document(doc)
            self.chunks.extend(chunked_docs)
            for chunked_doc in chunked_docs:
                chunk_embedding = self.embed_context(chunked_doc)
                self.faiss_index.add(chunk_embedding)
                self.chunk_index[chunk_counter] = doc_counter
                self.inverse_chunk_index[doc_counter].append(chunk_counter)
                chunk_counter += 1
        self.total_docs = len(self.documents)
        self.total_chunks = len(self.chunks)


    def embed_question(self, question: str):
        '''Embed the question in vector space with the question encoder.'''
        input_ids = self.question_tokenizer(
            question, return_tensors='pt')['input_ids']
        embeddings = self.question_model(
            input_ids).pooler_output.detach().numpy()
        return embeddings

    def embed_context(self, context: str):
        '''Embed the context (doc) in vector space with the question encoder.'''
        input_ids = self.context_tokenizer(
            context, return_tensors='pt')['input_ids']
        embeddings = self.context_model(
            input_ids).pooler_output.detach().numpy()
        return embeddings


    def search_index(self, question: str, k: int = 5):
        '''
        Search the vector index by encoding the question and then performing
        nearest neighbor on the FAISS index of context vectors.

        Args:
            question (str):
                The natural language question, e.g. `who is bill gates?`
            k (int):
                The number of documents to return from the index.
        '''
        if k > self.total_chunks:
            k = self.total_chunks
        question_embedding = self.embed_question(question)
        dists, chunk_ids = self.faiss_index.search(question_embedding, k=k)
        dists, chunk_ids = list(dists[0]), list(chunk_ids[0])
        dists = list(map(float, dists)) #For Flask
        structured_response = []
        for dist, chunk_id in zip(dists, chunk_ids):
            chunk = self.chunks[chunk_id]
            document_id = self.chunk_index[chunk_id]
            document = self.documents[document_id]
            blob = {
                'document': document,
                'document_id': document_id,
                'chunk': chunk,
                'chunk_id': int(chunk_id), #For Flask
                'faiss_dist': dist
            }
            structured_response.append(blob)
        return structured_response
