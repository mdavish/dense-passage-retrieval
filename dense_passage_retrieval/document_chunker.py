import re
import logging
from typing import List
from transformers import DPRReaderTokenizer
from functools import lru_cache


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    '''
    Override logging levels of different modules based on their name as a
    prefix. It needs to be invoked after the modules have been loaded so that
    their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is
          logging.ERROR
        - prefices: list of one or more str prefices to match (e.g.
          ["transformers", "torch"]). Optional. Default is `[""]` to match
           all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    '''
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


set_global_logging_level(logging.ERROR, prefices=["transformers"])


class DocumentChunker:

    '''
    Class for chunking up long documents into optimally large pieces so that
    they fit into BERT.
    '''

    tokenizer = DPRReaderTokenizer.from_pretrained(
        'facebook/dpr-reader-single-nq-base')
    MAX_TOKENS = 512
    MAX_TOKENS_QUESTION = 30
    MAX_TOKENS_DOCUMENT = MAX_TOKENS - MAX_TOKENS_QUESTION - 2  # [SEP] & [CLS]

    def __init__(self):
        pass


    @lru_cache()
    def get_token_length(self, string: str) -> int:
        '''Returns the number of WordPiece tokens that a string comprises.'''
        tokens = self.tokenizer.encode(string)
        return len(tokens)


    def chunk_document(self, document: str,
                       split_chars: List[str] = ['\n', '.'],
                       re_consolidate: bool = True,
                       ignore_overlong: bool = True) -> List[str]:
        '''
        Chunks up a long document into optimally large pieces so that they
        can be passed to BERT.

        Args:
            document (str):
                The string to be chunked up.
            split_chars (str):
                Which characters, in order, the document can be split on.
                Default is ['\n', '.'] to split on paragraphs then documents.s
            reconsolidate (bool):
                Whether or not to put the chunks back to gether such that they
                are as big as possible while still fitting into BERT.
            ignore_overlong (bool):
                Whether or not to ignore overlong pieces of text that cannot be
                chunked up to the appropriate size, given the split_chars.
        '''
        chunks = [document]
        chunk_lengths = list(map(self.get_token_length, chunks))
        for split_char in split_chars:
            new_chunks = []
            for chunk, length in zip(chunks, chunk_lengths):
                if length > self.MAX_TOKENS_DOCUMENT:
                    sub_chunks = chunk.split(split_char)
                    sub_chunks = [chunk for chunk in sub_chunks if chunk]
                    new_chunks.extend(sub_chunks)
                else:
                    new_chunks.append(chunk)
            chunks = new_chunks
            chunk_lengths = list(map(self.get_token_length, chunks))
        approved_chunks = []
        for chunk, length in zip(chunks, chunk_lengths):
            if length > self.MAX_TOKENS_DOCUMENT:
                if ignore_overlong:
                    logging.warning(f'Ignoring overlong string: {chunk}')
                else:
                    msg = f'''Cannot chunk overlong string:
                    {chunk[:100]}
                    Set ignore_overlong=True to ignore these errors.
                    '''
                    raise ValueError(msg)
            else:
                approved_chunks.append(chunk)
        if re_consolidate:
            lengths = list(map(self.get_token_length, approved_chunks))
            consolidated_chunks = []
            running_length = 0
            current_chunk = ''
            for chunk, length in zip(approved_chunks, lengths):
                if (running_length + length) < self.MAX_TOKENS_DOCUMENT:
                    current_chunk += chunk
                    running_length += length
                else:
                    consolidated_chunks.append(current_chunk)
                    current_chunk = chunk
                    running_length = length
            consolidated_chunks.append(current_chunk)
            return consolidated_chunks
        else:
            return approved_chunks
