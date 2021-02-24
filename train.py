"""
trains heuristic chess function on all games that were won by resign or mate
"""

from utils.model import model
import pandas as pd
import numpy as np
import chess
import matplotlib.pyplot as plt
from tensorflow import keras
from utils.chess_utils import *

df = pd.read_csv('data/games.csv', sep=';')
df = df[df['winner'] != 'draw']
df = df[df['victory_status'] != 'outoftime']
moves = df['moves'].values
winner = df['winner'].values
X = []
y = []

index = 0
for game in moves:
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
        rows = translate(matrix)
        X.append([rows])
        y.append(value)

    index += 1

# define X and y
X = np.array(X).reshape((len(X), 8, 8, 12))
y = np.array(y)

# start training
# set name of files
h5 = 'model.h5'
json = 'model.json'

# create callbacks
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

history = model.fit(X, y, epochs=1000, verbose=2, callbacks=callback)

# plot and save
plt.plot(history.history['loss'])
plt.savefig('history.png')
