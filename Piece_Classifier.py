import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import itertools
import BoardExtractor
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
import urllib.request

def get_model():
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(5,5), padding='valid',input_shape=(64,64,3),activation='relu' ))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid'))
    model.add(Conv2D(256, kernel_size=(3,3),strides=(1,1), padding='same',activation='relu' ))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid'))
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(12,activation='softmax'))
    return model

def get_fen_url(predictions):
    fen_not = {
        'B-Bishop':'b',
        'B-King':'k',
        'B-Knight':'n',
        'B-Pawn':'p',
        'B-Queen':'q',
        'B-Rook':'r',
        'W-Bishop':'B',
        'W-King':'K',
        'W-Knight':'N',
        'W-Pawn':'P',
        'W-Queen':'Q',
        'W-Rook':'R',
        'Empty':'1'
    }
    rows = []
    for i in range(8):
        row = ""
        for j in range(8):
            row += fen_not[predictions[i*8+j]]
        rows.append(row)
    rows = '/'.join(rows)
    fen_url = 'https://chessboardimage.com/{}.png'.format(rows)
    urllib.request.urlretrieve(fen_url,"result.png")
    return rows,fen_url

def show_results(filename,show=True):
    result_img = cv2.imread('result.png')
    fig = plt.figure(figsize=(12,12))

    fig.add_subplot(1,2,1)
    plt.imshow(cv2.imread(filename,0),cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Original Image')

    fig.add_subplot(1,2,2)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title('Prediction')

    fig.savefig('results.png',bbox_inches='tight')
    if show:
        plt.show()

def get_prediction(filename,show=True):
    model = get_model()
    model.load_weights('best_model_multi.h5')
    chessboard = BoardExtractor.get_chessboard(filename,show=False)
    chessboard = cv2.resize(chessboard,(512,512))
    squares = BoardExtractor.get_chessboard_squares(chessboard)
    labels = {0: 'B-Bishop', 1: 'B-King', 2: 'B-Knight', 3: 'B-Pawn', 4: 'B-Queen', 5: 'B-Rook', 6: 'W-Bishop', 7: 'W-King', 8: 'W-Knight', 9: 'W-Pawn', 10: 'W-Queen', 11: 'W-Rook'}
    predictions = []
    for i in range(64):
        img = squares[i]
        temp = img.copy()
        temp = BoardExtractor.preprocess_image(cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY))
        temp = temp[12:52,12:52]
        #check if the square is empty
        if np.sum(temp)/(40*40*255) < 0.2:
            predictions.append('Empty')
            continue
        pred = np.argmax(model.predict(np.array([img])),axis=1)
        predictions.append(labels[pred[0]])
    fen,link = get_fen_url(predictions)
    show_results(filename,show)
    return fen
