"""
test the heuristics the model predicts from board inputted as fen string
"""

import chess
from utils.chess_utils import *
from tensorflow import keras

# load the model
# first the json structure then the binary
model: keras.Sequential
with open('model.json', 'r') as json:
    model = keras.models.model_from_json(json.read())
model.load_weights('model.h5')

print('<< Enter a fen string, the network will evaluate the position. Press \'q\' to exit.')

while True:
    fen = input('>> ')

    if fen == 'q':
        break

    try:
        board = chess.Board(fen=fen)
    except ValueError:
        print('<< invalid fen string')
        continue

    matrix = make_matrix(board)
    rows = translate(matrix)

    print('<<', rows)
    print('<<',  model.predict([rows]))
