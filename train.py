from model import model
import pandas as pd
import numpy as np
import chess
import os
import matplotlib.pyplot as plt
from tensorflow import keras

df = pd.read_csv('data/games.csv', sep=';')
df = df[df['winner'] != 'draw']
moves = df['moves'].values[:100]
winner = df['winner'].values
X = []
y = []

chess_dict = {
    'p': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'P': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'n': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'N': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'b': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'B': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'r': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'q': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'k': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'K': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    '.': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}


def make_matrix(board):
    pgn = board.epd()
    foo = []
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        foo2 = []
        for thing in row:
            if thing.isdigit():
                for i in range(0, int(thing)):
                    foo2.append('.')
            else:
                foo2.append(thing)
        foo.append(foo2)
    return foo


def translate(matrix0, chess_dict0):
    rows0 = []
    for row in matrix0:
        terms = []
        for term in row:
            terms.append(chess_dict0[term])
        rows0.append(terms)
    return rows0


for game in moves:
    index = list(moves).index(game)
    all_moves = game.split()
    total_moves = len(all_moves)
    if winner[index] == 'black':
        game_winner = -1
    else:
        game_winner = 1
    board = chess.Board()
    for i in range(len(all_moves)):
        board.push_san(all_moves[i])
        value = game_winner * (i / total_moves)
        matrix = make_matrix(board.copy())
        rows = translate(matrix, chess_dict)
        X.append([rows])
        y.append(value)
X = np.array(X).reshape((len(X), 8, 8, 12))
y = np.array(y)

# start training
h5 = 'model.h5'
json = 'model.json'

checkpoint = keras.callbacks.ModelCheckpoint(h5,
                                             monitor='loss',
                                             verbose=0,
                                             save_best_only=True,
                                             save_weights_only=True,
                                             mode='auto',
                                             period=1)
es = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=500)
callback = [checkpoint, es]
model_json = model.to_json()

with open(json, "w") as json_file:
    json_file.write(model_json)

print('Training Network...')

history = model.fit(X, y, epochs=1000, verbose=2, callbacks=callback)

plt.plot(history.history['loss'])
plt.savefig('history.png')
