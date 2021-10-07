# Chess-Puzzle-Scanner

This project aims to create a tool to generate a link to engine analysis board from images of book chess puzzles.

## Board Extraction

The relevant functions are given in [BoardExtractor.py](https://github.com/SubarnoNR/Chess-Puzzle-Scanner/blob/cd5e23b34021712877d83e5f644d989917797fca/BoardExtractor.py) and the
visualisation for each image preprocessing step in [ExtractChessBoard.ipynb](https://github.com/SubarnoNR/Chess-Puzzle-Scanner/blob/4c8070aa652934ed5bb0e25a4517447cab4132df/ExtractChessBoard.ipynb)

Here is a result :

![Board Extractor](https://github.com/SubarnoNR/Chess-Puzzle-Scanner/blob/4c8070aa652934ed5bb0e25a4517447cab4132df/Images/extractedboard.png)

## Chess Pieces Classification

The model has been trained here, [ClassifierModel.ipynb](https://github.com/SubarnoNR/Chess-Puzzle-Scanner/blob/master/ClassifierModel.ipynb) achieving **97.5%** accuracy on various classic chess fonts.

Some results:

![Result1](https://github.com/SubarnoNR/Chess-Puzzle-Scanner/blob/4c8070aa652934ed5bb0e25a4517447cab4132df/result1.png)

![Result2](https://github.com/SubarnoNR/Chess-Puzzle-Scanner/blob/4c8070aa652934ed5bb0e25a4517447cab4132df/result2.png)

## Graphical User Interface

Here is how the final application looks like as of now :

![GUI Image](https://github.com/SubarnoNR/Chess-Puzzle-Scanner/blob/4c8070aa652934ed5bb0e25a4517447cab4132df/Images/GUI.png)

## Further Improvements 

- Train the model on more chess fonts including chess.com and lichess piece sets.
- Improve the aesthetics of the GUI application
