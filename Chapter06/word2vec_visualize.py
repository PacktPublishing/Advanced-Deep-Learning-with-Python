import logging
import pprint  # beautify prints

import gensim.downloader as gensim_downloader
import matplotlib.pyplot as plt
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO)

# Show all available datasets
pprint.pprint(gensim_downloader.info())

# Train the model
model = Word2Vec(
    sentences=gensim_downloader.load('text8'),  # download and load the text8 dataset
    sg=0, size=100, window=5, negative=5, min_count=5, iter=5)

pprint.pprint(model.wv.most_similar(positive=['woman', 'king'], negative=['man']))

# Collect the words (and vectors), most similar to the words below
target_words = ['mother', 'car', 'tree', 'science', 'building', 'elephant', 'green']
word_groups, embedding_groups = list(), list()

for word in target_words:
    words = [w for w, _ in model.most_similar(word, topn=5)]
    word_groups.append(words)

    embedding_groups.append([model.wv[w] for w in words])

# Train the t-SNE algorithm
embedding_groups = np.array(embedding_groups)
m, n, vector_size = embedding_groups.shape
tsne_model_en_2d = TSNE(perplexity=8, n_components=2, init='pca', n_iter=5000)

# generate 2d embeddings from the original 100d ones
embeddings_2d = tsne_model_en_2d.fit_transform(embedding_groups.reshape(m * n, vector_size))
embeddings_2d = np.array(embeddings_2d).reshape(m, n, 2)

# Plot the results
plt.figure(figsize=(16, 9))
# Different color and marker for each group of similar words
color_map = plt.get_cmap('Dark2')(np.linspace(0, 1, len(target_words)))
markers = ['o', 'v', 's', 'x', 'D', '*', '+']

# Iterate over all groups
for label, similar_words, emb, color, marker in \
        zip(target_words, word_groups, embeddings_2d, color_map, markers):
    x, y = emb[:, 0], emb[:, 1]

    # Plot the points of each word group
    plt.scatter(x=x, y=y, c=color, label=label, marker=marker)

    # Annotate each vector with it's corresponding word
    for word, w_x, w_y in zip(similar_words, x, y):
        plt.annotate(word, xy=(w_x, w_y), xytext=(0, 15),
                     textcoords='offset points', ha='center', va='top', size=10)
plt.legend()
plt.grid(True)
plt.show()
