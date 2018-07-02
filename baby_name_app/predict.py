import numpy
import os
from keras.utils import np_utils
from keras import backend as K

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate():
    # load ascii text and covert to lowercase
    filename = os.path.join(BASE_DIR, 'baby_name_app/boy.txt')
    raw_text = open(filename).read()
    raw_text = raw_text.lower()

    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))

    n_chars = len(raw_text)
    n_vocab = len(chars)

    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 20
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append([char_to_int[seq_out]])
    n_patterns = len(dataX)

    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    # Load model
    from keras.models import load_model
    model = load_model(os.path.join(BASE_DIR, 'baby_name_app/model-50-epochs.h5'))

    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # pick a random seed
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print("Seed:")
    print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

    # generate characters
    characters = []
    for i in range(30):
    	x = numpy.reshape(pattern, (1, len(pattern), 1))
    	x = x / float(n_vocab)
    	prediction = model.predict(x, verbose=0)
    	index = numpy.argmax(prediction)
    	result = int_to_char[index]
    	seq_in = [int_to_char[value] for value in pattern]
    	characters.append(result)
    	pattern.append(index)
    	pattern = pattern[1:len(pattern)]

    words = []
    word_single = ""
    for ch in characters:
        if(ch == "\n"):
            if(len(word_single) >= 3):
                words.append(word_single)
            word_single = ""
        else:
            word_single += ch

    K.clear_session()
    return words
