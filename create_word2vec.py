# -*- coding: UTF-8 -*-

from __future__ import print_function
import gensim
from utils import load_text


class VectorCreator:

    def __init__(self, source, bigrams=False):
        """
        Construct a vector creators class to generate the word2vec vectors for a given source text
        :param source: The text to get the word2vec from
        :param bigrams: Whether or not to use bigrams
        """
        # Read in the source text file
        print("Loading in text")
        text = load_text(source)

        # Parse the text in the file
        parsed_words = [words.split(' ') for words in [sentences for sentences in text.split('.')]]

        # If we want to use bigrams
        if bigrams:
            bigram_parsed_words = gensim.models.Phrases(parsed_words)
            print("Training word2vec model on file: " + source)
            w2v = gensim.models.Word2Vec(bigram_parsed_words, size=300, min_count=1,
                                         iter=10, window=8, sg=1, hs=0, negative=5)

        # Use only unigrams -- cheaper training
        else:
            print("Training word2vec model on file: " + source)
            w2v = gensim.models.Word2Vec(parsed_words, size=300, min_count=1,
                                         iter=10, window=8, sg=1, hs=0, negative=5)

        print("Saving word2vec model")
        w2v.save('vectors.bin')


if __name__ == "__main__":
    VectorCreator("the_republic.txt")

