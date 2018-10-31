# '''Trains a memory network on each of the 2 bAbI datasets.
# References:
# - Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
#   "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
#   http://arxiv.org/abs/1502.05698
# - Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
#   "End-To-End Memory Networks",
#   http://arxiv.org/abs/1503.08895
# '''

#run pip install -r requirements.txt or pip3 install -r requirements.txt
#to install dependencies

from __future__ import print_function
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re
import os
from random import shuffle


# configure challenge type here
challenge_type = 'single_supporting_fact_10k'
# challenge_type = 'two_supporting_facts_10k'

# configure epoch here
# in 'single_supporting_fact_10k', if epochs != 1, epochs = 120
# in 'two_supporting_facts_10k', if epochs != 1, epochs = 40
epochs = 40

# when using Jupyter Notebook
dir = os.getcwd()
# when using local runtime
dir = os.path.dirname(__file__)

batch_size = None
dropout = 0.3


def tokenize(sent):
    return [x.strip() for x in re.split(r'(\W+)?', sent) if x.strip()]

# can also use nltk
# from nltk.tokenize import word_tokenize
# import nltk
# nltk.download('punkt')
# def tokenize(sent):
#     return word_tokenize(sent)


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format'''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            #If only_supporting is true, only the sentences
            #that support the answer are kept.
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else: 
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories'''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)

    def flatten(data): return reduce(lambda x, y: x + y, data)
    # convert the sentences into a single story
    # If max_length is supplied, any stories longer than max_length tokens will be discarded.
    data = [(flatten(story), q, answer) for story, q,
            answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))


try:
    path = get_file('babi-tasks-v1-2.tar.gz',
                    origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
tar = tarfile.open(path).extractfile

challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
}

challenge = challenges[challenge_type]

print('Extracting stories for the challenge:', challenge_type)
train_stories = get_stories(tar(challenge.format('train')))
test_stories = get_stories(tar(challenge.format('test')))
shuffle(train_stories)
shuffle(test_stories)

vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input,  , answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

inputs_train, queries_train, answers_train = vectorize_stories(train_stories,
                                                               word_idx,
                                                               story_maxlen,
                                                               query_maxlen)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories,
                                                            word_idx,
                                                            story_maxlen,
                                                            query_maxlen)

print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')
print('Building model...')


def one_supporting_facts(epochs=epochs, batch_size=batch_size,
                         dropout=dropout, output_dim=64,
                         LSTM_unit=32):
    
    input_sequence = Input((story_maxlen,))
    question = Input((query_maxlen,))

    input_encoder_m = Sequential()
    input_encoder_m.add(Embedding(input_dim=vocab_size,
                                  output_dim=output_dim))
    input_encoder_m.add(Dropout(dropout))
    
    input_encoder_c = Sequential()
    input_encoder_c.add(Embedding(input_dim=vocab_size,
                                  output_dim=query_maxlen))
    input_encoder_c.add(Dropout(dropout))
    
    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=vocab_size,
                                   output_dim=output_dim,
                                   input_length=query_maxlen))
    question_encoder.add(Dropout(dropout))
    
    input_encoded_m = input_encoder_m(input_sequence)
    input_encoded_c = input_encoder_c(input_sequence)
    question_encoded = question_encoder(question)

    match = dot([input_encoded_m, question_encoded], axes=(2, 2))
    match = Activation('softmax')(match)
    response = add([match, input_encoded_c])
    response = Permute((2, 1))(response)
    answer = concatenate([response, question_encoded])
    answer = LSTM(LSTM_unit)(answer)  

    answer = Dropout(dropout)(answer)
    answer = Dense(vocab_size)(answer)  
    answer = Activation('softmax')(answer)
    model = Model([input_sequence, question], answer)
    return model


def two_supporting_facts(epochs=epochs, batch_size=batch_size,
                         dropout=dropout, embed_hidden_size=50,
                         sent_hidden_size=100,
                         query_hidden_size=100):
    sentence = layers.Input(shape=(story_maxlen,), dtype='int32')
    encoded_sentence = layers.Embedding(
        vocab_size, embed_hidden_size)(sentence)
    encoded_sentence = layers.Dropout(dropout)(encoded_sentence)

    question = layers.Input(shape=(query_maxlen,), dtype='int32')
    encoded_question = layers.Embedding(
        vocab_size, embed_hidden_size)(question)
    encoded_question = layers.Dropout(dropout)(encoded_question)
    encoded_question = LSTM(embed_hidden_size)(encoded_question)
    encoded_question = layers.RepeatVector(story_maxlen)(encoded_question)

    merged = layers.add([encoded_sentence, encoded_question])
    merged = LSTM(embed_hidden_size)(merged)
    merged = layers.Dropout(dropout)(merged)
    preds = layers.Dense(vocab_size, activation='softmax')(merged)
    model = Model([sentence, question], preds)
    return model


if challenge_type == 'single_supporting_fact_10k':
    if epochs != 1:
        epochs = 120
    # filepath = os.path.join(dir, 'one_fact_chatbot_model.h5')
    filepath = os.path.join(dir, 'one_fact_chatbot_model_weight.h5')
    log_dir = os.path.join(dir, 'logs/1')
    output_dim = 64
    LSTM_unit = 32
    model = one_supporting_facts(epochs=epochs, batch_size=batch_size,
                                 dropout=dropout, output_dim=output_dim,
                                 LSTM_unit=LSTM_unit)
else:
    if epochs != 1:
        epochs = 40
    filepath = os.path.join(dir, 'two_facts_chatbot_model_weight.h5')
    log_dir = os.path.join(dir, 'logs/2')
    embed_hidden_size = 20
    sent_hidden_size = 100
    query_hidden_size = 100
    model = two_supporting_facts(dropout=dropout, batch_size=batch_size,
                                 epochs=epochs,
                                 embed_hidden_size=embed_hidden_size,
                                 sent_hidden_size=sent_hidden_size,
                                 query_hidden_size=query_hidden_size)


model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])


# make the file directories needed
new_dir = os.path.join(dir, 'logs')
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# # train
# print('Training...')
# callbacks = [
#     callbacks.ModelCheckpoint(filepath=filepath, verbose=1,
#                               monitor='val_loss', save_best_only=True),
#     # Check out the train history later with Tensorboard
#     callbacks.TensorBoard(log_dir=log_dir),
#     callbacks.EarlyStopping(patience=20)]

# model.fit([inputs_train, queries_train], answers_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_split=0.1,
#           callbacks=callbacks)

# del model
# model = load_model(filepath=filepath)
model.load_weights(filepath=filepath)

# evaluate
test_result = model.evaluate([inputs_test, queries_test], answers_test)
print(
    f'Test result:\nTest loss = {test_result[0]}, Test accuracy = {test_result[1]}')

# predict
predictions = model.predict([inputs_test, queries_test])
re_word_idx = {v: k for k, v in word_idx.items()}

for i in range(9):
    for j, k in enumerate(test_stories[i]):
        if j < 2:
            print('\n' + ' '.join(k))
        if j == 2:
            print('\nGround truth: ' + ''.join(k))
    predicted = re_word_idx[np.argmax(predictions[i])]
    print(f'Prediction  : {predicted}\n')
