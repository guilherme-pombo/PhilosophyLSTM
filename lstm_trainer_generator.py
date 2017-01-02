# -*- coding: UTF-8 -*-

import gensim as gs
import numpy as np
import lstm_model
from utils import one_hot, load2, sample
import random

sd_len = 12

x_D = []
y_D = []
v_D = []
i_D = []

word_coding = {}
coded_word = {}
coded_vector = []
vec_values = {}


class ModelTrainer:
    """
    Use this class to initialize the word vectors and then to train the LSTM model
    and/or generate sentences given a trained LSTM model
    """

    model_trained = False

    def __init__(self, source):
        print("Loading in word2vec model")
        self.vmodel = gs.models.Word2Vec.load('vectors.bin')

        print("Loading in text")
        text = load2(source)
        parsed_words = text.split(" ")

        code_num = 1

        print("Creating word -> vector dictionary...")
        for word in parsed_words:
            if not word in word_coding:
                word_coding[word] = code_num
                coded_word[code_num] = word
                code_num += 1
                vec_values[word] = self.vmodel[word]
            coded_vector.append(word_coding[word])
        print('Number of distinct words: ', len(word_coding))

        sd_size = int(len(coded_vector) / sd_len)

        x_d = y_d = v_d = i_d = []

        for idx in range(0, sd_size - 1):
            for iidx in range(0, sd_len - 1):
                indexD = coded_vector[idx * sd_len + iidx + 0:(idx + 1) * sd_len + iidx]
                i_D.append(indexD)

                vectorValD = [vec_values[myWord] for myWord in
                              parsed_words[idx * sd_len + iidx + 0:(idx + 1) * sd_len + iidx]]
                x_D.append(vectorValD)
                y_D.append(one_hot(coded_vector[(idx + 1) * sd_len + iidx], word_coding))
                v_D.append(vec_values[parsed_words[(idx + 1) * sd_len + iidx]])

        self.x_d = np.asarray(x_D)
        self.y_d = np.asarray(y_D)
        self.v_d = np.asarray(v_D)
        self.i_d = np.asarray(i_D)

        print('shapes: ' + str(self.x_d.shape))

    def train_model(self, epochs=20):
        """
        Train the LSTM model
        :param epochs:
        :return:
        """
        self.model = lstm_model.create_model(word_coding)

        print("Training model for " + str(epochs) + " epochs")

        for epoch in range(0, epochs):
            print("Epoch: " + str(epoch))
            for j in range(2):
                self.model.fit({'input': self.x_d, 'output1': self.y_d}, nb_epoch=5)
                self.model.save_weights('lstm-weights', overwrite=True)

            preds = self.model.predict({'input': self.x_d[:5000]}, verbose=0)
            train_accuracy = np.mean(np.equal(np.argmax(self.y_d[:5000], axis=-1),
                                              np.argmax(preds['output1'][:5000], axis=-1)))

            print("Model accuracy: " + str(train_accuracy))

        ModelTrainer.model_trained = True

    def generate_sentences(self, len_sentences=60, load=True):
        """
        Generate sentences given a trained LSTM model
        :param len_sentences: the length of the sentences to be generated
        :param load: whether to load in a model, or to use the trained one, only works if you ran train_model before
        :return:
        """

        model = lstm_model.create_model(word_coding)

        if load:
            model.load_weights('lstm-weights')

        else:
            if not ModelTrainer.model_trained:
                raise Exception("The model hasn't been trained. Either train it or load in the weights from a file")
            model = self.model

        seedSrc = i_D
        outSentences = []
        while len(outSentences) < len_sentences:
            start_index = random.randint(0, len(seedSrc) - 1)
            sentence = seedSrc[start_index: start_index + 1]

            sentOutput = ''

            for iteration in range(500):
                vecsentence = []
                for vcode in sentence[0]:
                    vecsentence.append(self.vmodel[coded_word[vcode]])
                vecsentence = np.reshape(vecsentence, (1, len(vecsentence), 300))
                preds = model.predict({'input': vecsentence}, verbose=0)['output1'][0]
                next_index = sample(preds, 0.2)
                if next_index in coded_word:
                    next_char = coded_word[next_index]
                    sentence = np.append(sentence[0][1:], [next_index]).reshape(np.asarray(sentence).shape)
                    sentOutput += next_char + ' '
            print(sentOutput)


if __name__ == "__main__":
    mt = ModelTrainer("the_republic.txt")
    mt.train_model(20)
    mt.generate_sentences(len_sentences=50, load=False)

