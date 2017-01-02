import numpy as np

"""
File that contains various utility functions
"""


def sample(a, temperature=1.0):
    """
    Sample an index from a list of probabilities
    :param a:
    :param temperature:
    :return:
    """
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


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
