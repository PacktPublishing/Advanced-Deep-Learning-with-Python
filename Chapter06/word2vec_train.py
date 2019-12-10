import logging
import pprint  # beautify prints

import gensim
import nltk
nltk.download('punkt')

logging.basicConfig(level=logging.INFO)


class TokenizedSentences:
    """Split text to sentences and tokenize them"""

    def __init__(self, filename: str):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as f:
            corpus = f.read()

        raw_sentences = nltk.tokenize.sent_tokenize(corpus)
        for sentence in raw_sentences:
            if len(sentence) > 0:
                yield gensim.utils.simple_preprocess(sentence, min_len=2, max_len=15)


sentences = TokenizedSentences('war_and_peace.txt')

model = gensim.models.word2vec. \
    Word2Vec(sentences=sentences,
             sg=1,  # 0 for CBOW and 1 for Skip-gram
             size=100,  # size of the embedding vector
             window=5,  # the size of the context window
             negative=5,  # negative sampling word count
             min_count=5,  # minimal word occurrences to include
             iter=5,  # number of epochs
             )

print("Words most similar to 'mother':")
pprint.pprint(model.wv.most_similar(positive='mother', topn=5))

print("Words most similar to 'woman' and 'king':")
pprint.pprint(model.wv.most_similar(positive=['woman', 'king'], topn=5))
