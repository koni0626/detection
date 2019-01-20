# coding:UTF-8
import os
import numpy
import glob
import conf
import cv2
import numpy as np
import random
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import Model, Input, Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Flatten, Dense, Activation
#from keras.layers.merge import add, multiply, concatenate, maximum, average
#from keras.layers.core import Activation, Flatten, Reshape
from keras.optimizers import Adam


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2.*intersection + 1) / (K.sum(y_true) + K.sum(y_pred) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
    
    
def my_yolov1():
    output_size = conf.GRID_SIZE[0] * conf.GRID_SIZE[1] * conf.CLASS_NUM * 5

    model = Sequential()
    # 畳み込み層追加
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation="relu", input_shape=(3, conf.IMAGE_SIZE[0], conf.IMAGE_SIZE[1])))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))


    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))

    model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))


    model.add(Flatten())

    model.add(Dense(4096, activation="relu"))
    model.add(Dense(4096, activation="relu"))

    model.add(Dense(output_size, activation="sigmoid"))
    
    return model


def coord_read(file_name):
    data = np.zeros(conf.GRID_SIZE[0] * conf.GRID_SIZE[1] * conf.CLASS_NUM * 5)
    grid_x_delims = 1. / conf.GRID_SIZE[0]
    grid_y_delims = 1. / conf.GRID_SIZE[1]
    
    with open(file_name) as f:
        records = f.readlines()
        for record in records:
           # print(record)
            d = record.split(" ")
            c = int(d[0])
            c_x = float(d[1])
            c_y = float(d[2])
            w = float(d[3])
            h = float(d[4])
            
            x_grid = int(c_x / grid_x_delims)
            y_grid = int(c_y / grid_y_delims)
            pos = (y_grid * conf.GRID_SIZE[0] + x_grid) * conf.CLASS_NUM * 5
            pos = pos + c * 5
            #print(pos)
            data[pos] = 1.
            data[pos + 1] = c_x
            data[pos + 2] = c_y
            data[pos + 3] = w
            data[pos + 4] = h

    return data
        
        
def data_generator(batch_size):

    search_path = os.path.join(conf.TRAIN_DATA_PATH, "*.txt")
    train_x_src_list = []
    train_y_src_list = []
    file_list = glob.glob(search_path)
    for txt in file_list:
        uuid_name = txt.replace("\\", "/").split("/")[-1]
        uuid_name = uuid_name.split(".")[-2]
        train_x_src_list.append(os.path.join(conf.TRAIN_DATA_PATH, uuid_name + ".jpg"))
        train_y_src_list.append(os.path.join(conf.TRAIN_DATA_PATH, uuid_name + ".txt"))

    
    count = 0
    i = 0
    data_size = len(train_x_src_list)
    while True:
        train_x_list = []
        train_y_list = []
        for b in range(batch_size):
            i = random.randint(0, data_size - 1)
            img = cv2.imread(train_x_src_list[i])
            img = cv2.resize(img, conf.IMAGE_SIZE).T/255.
            train_x_list.append(img)
            
            y = coord_read(train_y_src_list[i])
            train_y_list.append(y)
            
        
        yield np.array(train_x_list), np.array(train_y_list)
    

if __name__ == '__main__':
    model = my_yolov1()
    OPT = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.)
    model.compile(optimizer=OPT,loss="mse",metrics=["accuracy"])
    model.load_weights("../weight/yolov1.108-0.00-0.86.hdf5")
    cp_cb = ModelCheckpoint(filepath=conf.SAVE_WEIGHT_FILE, monitor='acc', verbose=1, save_best_only=True, mode='auto')
    cp_his = CSVLogger(conf.LOG_FILE)
    model.fit_generator(data_generator(8),
                        steps_per_epoch=1550/4, epochs=1000, callbacks=[cp_cb, cp_his])

