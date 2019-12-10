"""This example is partially based on https://github.com/pytorch/tutorials/blob/master/intermediate_source/seq2seq_translation_tutorial.py"""

from __future__ import unicode_literals, print_function, division

import random
import re
import unicodedata
from io import open

import torch

GO_token = 0
EOS_token = 1


class Lang:
    """Class that represents the vocabulary in one language"""

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "GO", 1: "EOS"}
        self.n_words = 2  # Count GO and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1


MAX_LENGTH = 10


class NMTDataset(torch.utils.data.Dataset):
    """
    Dataset for NMT
    The output is a tuple of two tensors, which represent the same sequence in the source and target languages
    Each sentence tensor contains the indices of the words in the vocabulary
    """

    def __init__(self, txt_file, dataset_size: int):
        lines = open(txt_file, encoding='utf-8'). \
            read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [[self._normalize_string(s) for s in l.split('\t')] for l in lines]
        pairs = [list(reversed(p)) for p in pairs]

        self.input_lang = Lang('fra')
        self.output_lang = Lang('eng')

        # Filter the pairs to reduce the size of the dataset
        filtered = list()
        eng_prefixes = (
            "i am ", "i m ",
            "he is", "he s ",
            "she is", "she s ",
            "you are", "you re ",
            "we are", "we re ",
            "they are", "they re "
        )

        for p in pairs:
            if len(p[0].split(' ')) < MAX_LENGTH and \
                    len(p[1].split(' ')) < MAX_LENGTH and \
                    p[1].startswith(eng_prefixes):
                filtered.append(p)

        pairs = filtered
        pairs = [random.choice(pairs) for _ in range(dataset_size)]

        # Create vocabularies
        for pair in pairs:
            self.input_lang.addSentence(pair[0])
            self.output_lang.addSentence(pair[1])

        self.pairs = pairs

        self.dataset = list()

        # Convert all sentences to tensors
        for pair in self.pairs:
            source_sentence_by_index = \
                [self.input_lang.word2index[word] for word in pair[0].split(' ')] + [EOS_token]

            output_sentence_by_index = \
                [self.output_lang.word2index[word] for word in pair[1].split(' ')] + [EOS_token]

            source_sentence_tensor = torch.tensor(source_sentence_by_index, dtype=torch.long).view(-1, 1)
            output_sentence_tensor = torch.tensor(output_sentence_by_index, dtype=torch.long).view(-1, 1)

            self.dataset.append((source_sentence_tensor, output_sentence_tensor))

    def _normalize_string(self, s):
        s = ''.join(
            c for c in unicodedata.normalize('NFD', s.lower().strip())
            if unicodedata.category(c) != 'Mn'
        )
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def sentence_to_sequence(self, sentence: str):
        """Convert the string sentence to a tensor"""
        sequence = [self.input_lang.word2index[word] for word in self._normalize_string(sentence).split(' ')] + [EOS_token]
        sequence = torch.tensor(sequence, dtype=torch.long).view(-1, 1)
        return sequence

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
