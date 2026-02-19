#!/bin/bash

# Download GloVe
if [ ! -f "glove.6B.100d.txt" ]; then
    echo "Downloading GloVe..."
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip glove.6B.100d.txt
    rm glove.6B.zip
else
    echo "GloVe embeddings found."
fi

# Download FastText
if [ ! -f "wiki-news-300d-1M-subword.vec" ]; then
    echo "Downloading FastText..."
    # now using subword embeddings
    curl -O https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip
    unzip wiki-news-300d-1M-subword.vec.zip
    rm wiki-news-300d-1M-subword.vec.zip
else
    echo "FastText embeddings found."
fi

echo "Embeddings loaded."
