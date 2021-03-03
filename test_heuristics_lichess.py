"""
test the heuristics the model predicts from board inputted as fen string
"""

import chess
from tensorflow import keras

# utility functions
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


def translate(matrix):
    rows = []
    for row in matrix:
        terms = []
        for term in row:
            terms.append(chess_dict[term])
        rows.append(terms)
    return rows


model_name = input('enter model name:')

# load the model
# first the json structure then the binary
model: keras.Sequential
with open('models/' + model_name + '.json', 'r') as json:
    model = keras.models.model_from_json(json.read())
model.load_weights('models/' + model_name + '.h5')

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

    print('<<', matrix)
    print('<<', model.predict([rows])[0][0])
