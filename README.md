# Support Chess Bot with Deep Learning
- creates a heuristic function for chess game, can be implemented with the [minimax algorithm](https://en.wikipedia.org/wiki/Minimax) (and alpha beta pruning)
- using [Chess Game Dataset (Lichess)](https://www.kaggle.com/datasnaek/chess) [20000 games]
- install requirements.txt or for gpu usage install `tensorflow-gpu` ([how to use](https://stackoverflow.com/questions/51306862/how-do-i-use-tensorflow-gpu))
- I found [this](https://towardsdatascience.com/creating-a-chess-engine-with-deep-learning-b9477ff3ee3d) post to be pretty helpful.
- "Keep in mind that the scale goes between -1 and 1, -1 being a checkmate for black and 1 being a checkmate for white."
- you can use the notebook to train the model with [Google Colab](https://colab.research.google.com/)