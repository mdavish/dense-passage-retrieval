import torch
from typing import List
from transformers import DPRReader, DPRReaderTokenizer
from .document_chunker import DocumentChunker


class DPRReader(DocumentChunker):

    '''
    Class for "reading" retrieved documents with DPR, which performs two
    functions: re-ranking them and providing candidate answers to the question.
    '''

    reader_tokenizer = DPRReaderTokenizer.from_pretrained(
        'facebook/dpr-reader-single-nq-base')
    reader_model = DPRReader.from_pretrained(
        'facebook/dpr-reader-single-nq-base', return_dict=True)


    def __init__(self):
        super(DocumentChunker).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            self.reader_model = self.reader_model.cuda()

    def _reconstruct_tokens(self, bert_tokens: List[str]):
        '''
        Utility function for reassembling WordPiece tokens into
        human-readable strings.
        '''
        output_string = ''
        for token in bert_tokens:
            if token[:2] == '##':
                output_string += token[2:]
            else:
                output_string += ' '
                output_string += token
        return output_string[1:]

    def read_documents(self, question: str, documents: List[str],
                       titles: List[str]):
        '''
        Reads a series of `documents` and `titles` and rates their relevance
        to the `question` as well as proposes an answer.

        Args:
            question (str):
                The question string (e.g. `who is bill gates?`)
            documents (List[str]):
                List of documents to rate/propose an answer from.
            titles (List[str]):
                List of the titles of those documents
        '''
        assert len(documents) == len(titles)
        encoded_inputs = self.reader_tokenizer(
            questions=question,
            titles=titles,
            texts=documents,
            return_tensors='pt',
            padding=True
        )
        input_ids = encoded_inputs['input_ids']
        encoded_inputs = encoded_inputs.to(self.device)
        outputs = self.reader_model(**encoded_inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        relevance_logits = outputs.relevance_logits
        responses = []
        for i in range(len(documents)):
            title = titles[i]
            document = documents[i]
            start = start_logits[i]
            end = end_logits[i]
            relevance = relevance_logits[i]
            inp_ids = input_ids[i]
            input_tokens = self.reader_tokenizer.convert_ids_to_tokens(inp_ids)
            answer_start = int(start.argmax())
            answer_end = int(end.argmax())
            relevance = float(relevance.max())
            answer_tokens = input_tokens[answer_start : answer_end + 1]
            answer_str = self._reconstruct_tokens(answer_tokens)
            response = {
                'answer': answer_str,
                'relevance': relevance,
                'title': title,
                'document': document
            }
            responses.append(response)
        response = responses.sort(key=lambda x: -x['relevance'])
        return responses

    def read_chunked_document(self, question: str, document: str, title: str):
        '''
        Read a single document that may be exceed the maximum length BERT
        can handle, so chunk it up into pieces.

        For args see DPRReader.read_documents()
        '''
        chunked_docs = self.chunk_document(document)
        titles_list = [title for i in range(len(chunked_docs))]
        return self.read_documents(question, chunked_docs, titles_list)
