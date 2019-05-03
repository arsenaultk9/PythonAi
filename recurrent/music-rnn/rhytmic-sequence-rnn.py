import json
import numpy as np
import random
import sys

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop

maxlen = 31

with open("encoding.json", "r") as read_file:
    data = json.load(read_file)

X = data['X']
Y = data['Y']

note_equivalents = {
    0: "#",
    1: "S",
    2: "E",
    3: "E.",
    4: "Q",
    5: "Q.",
    6: "H",
    7: "H.",
    8: "W",
}

print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(31, 9), name='input'))
model.add(Dense(9, activation='softmax', name='ouput'))

optimizer = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def one_hot_encoding_to_music_sequence(segment):
    text = ''
    for row in segment:
        for index in range(len(row)):
            if(row[index] == True):
                text += note_equivalents[index]

    return text


# music_sequences = []
# for segment in X:
#     music_sequence = one_hot_encoding_to_music_sequence(segment)
#     music_sequences.append(music_sequence)

# quarter_note_sequences = []
# index = 0
# indices_to_remove = []

# for sequence in music_sequences:
#     quarter_or_silence_count = 0
#     for sequence_char in sequence:
#         if(sequence_char == '#' or sequence_char == 'Q'):
#             quarter_or_silence_count += 1

#     if(quarter_or_silence_count >= 29):
#         quarter_note_sequences.append(sequence)
#         indices_to_remove.append(index)

#     index += 1

# indices_to_remove.reverse()
# for indice in indices_to_remove:
#     del X[indice]
#     del Y[indice]


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

    # preds = np.asarray(preds).astype('float64')
    # return np.argmax(preds)


X = np.array(X)
Y = np.array(Y)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(X) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        segment = X[start_index]
        generated += one_hot_encoding_to_music_sequence(segment) + ' | '
        sys.stdout.write(generated)

        for i in range(64):
            x_pred = np.array([segment])

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_note = note_equivalents[next_index]

            generated += next_note

            next_note_arr = np.zeros(len(note_equivalents))
            next_note_arr[next_index] = True
            segment = np.append(segment[1:], [next_note_arr], axis=0)

            sys.stdout.write(next_note)
            sys.stdout.flush()
        print()


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(X, Y,
          batch_size=128,
          epochs=240,
          shuffle=False,
          callbacks=[print_callback])

model.save('rhytmic_sequence_model.h5')
# tf.keras.models.save_model(model, 'model.h5')
