import numpy as np

"""
File that contains various utility functions
"""


def sample(probs, temperature=1.0):
    """
    Sample an index from a list of probabilities
    Decreasing the temperature from 1 to some lower number (e.g. 0.5) makes the RNN more confident,
    but also more conservative in its samples. Conversely, higher temperatures will give more diversity
    but at cost of more mistakes (e.g. spelling mistakes, etc)
    :param a:
    :param temperature:
    :return:
    """
    probs = np.log(probs) / temperature
    probs = np.exp(probs) / np.sum(np.exp(probs))
    return np.argmax(np.random.multinomial(1, probs, 1))


# train the model, output generated text after each iteration
def get_sentence(wordVec, codedWord):
    sent = ''
    for wordVal in wordVec:
        sent += codedWord[wordVal] + ' '
    return sent


def one_hot(index, wordCoding):
    """
    Generate a one hot vector it's of type [0,0,0,1,0]
    :param index:
    :param wordCoding:
    :return:
    """
    retVal = np.zeros((len(wordCoding)), dtype=np.bool)
    retVal[index] = 1
    return retVal


def normalizeVector(vecs):
    retval = {}
    tempval = []
    for vkey in vecs:
        tempval.append(vecs[vkey])
    vecMean = np.mean(np.asarray(tempval), axis=0)
    vecStd = np.std(np.asarray(tempval), axis=0)
    for veckey in vecs:
        retval[veckey] = (vecs[veckey] - vecMean) / vecStd
    return retval


def load_text(source):
    """
    Use this to load the text when training the word2vec model
    :param source:
    :return:
    """
    text = open(source).read().lower()
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace("\"", "")
    text = text.replace("\'", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("e.g.", "example")
    text = text.replace("...", " ")
    text = text.replace(" - ", ", ")
    text = text.replace(" t ", " to ")
    text = text.replace(". ", " [dot]. ")
    text = text.replace("! ", " [xcm]. ")
    text = text.replace("? ", " [q]. ")
    text = text.replace(", ", " [comma] ")
    text = text.replace(" $", " [dlr] ")
    text = text.replace(": ", " [cln] ")
    text = text.replace("; ", " [scln] ")
    text = text.replace("% ", " [pcnt] ")
    text = text.replace(". ", ".")
    text = text.replace("   ", " ")
    text = text.replace("  ", " ")
    return text


def load2(source):
    """
    Use this to load the text when training the LSTM model
    :param source:
    :return:
    """
    text = open(source).read().lower()
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace("\"", "")
    text = text.replace("\'", "")
    text = text.replace("e.g.", "example")
    text = text.replace("...", " ")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace(" - ", ", ")
    text = text.replace(".. ", ". ")
    text = text.replace("! ", " [xcm] ")
    text = text.replace(". ", " [dot] ")
    text = text.replace("? ", " [q] ")
    text = text.replace(", ", " [comma] ")
    text = text.replace(" $", " [dlr] ")
    text = text.replace(": ", " [cln] ")
    text = text.replace("; ", " [scln] ")
    text = text.replace("% ", " [pcnt] ")
    text = text.replace("   ", " ")
    text = text.replace("  ", " ")

    return text
