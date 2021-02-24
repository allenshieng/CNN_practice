import torch
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D


def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    plt.imshow(image, cmap='binary')
    plt.show()

def plot_images_labels_prediction(images,labels,prediction,idx,num=10):

    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num > 25 : num=25
    for i in range(0,num):
        ax = plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap = 'binary')
        title = "label=" + str(labels[idx])
        if len(prediction) > 0:
            title += ",predict="+str(prediction[idx])

        ax.set_title(title,fontsize = 10) #設定title大小
        ax.set_xticks([]);ax.set_yticks([]) #把x和y軸去掉
        idx += 1
    plt.show()


#載入Mnist手寫辨識資料
# keras.datasets.mnist.load_data()

#Training的資料有60000筆而Testing的資料有10000筆都是28*28 pixel大小的手寫數字圖片
#x_train : images
#y_train : labels
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# print('x_train_image:',x_train.shape)
# print('y_train_label:',y_train.shape)

# print('x_test_image:',x_test.shape)
# print('y_test_label:',y_test.shape)

# #畫出圖片長相以及Label
# plot_image(x_train[0])
# plot_image(x_train[1])
# print(y_train[0])
# print(y_train[1])

# plot_images_labels_prediction(x_train,y_train,[],0,10)
# plot_images_labels_prediction(x_test,y_test,[],0,10)

# print(x_train.shape)

#由於圖片通常是RGB三個顏色所組成的，假設圖片大小是28*28的彩色圖片，實際上的資料維度就是28*28*3。不過這邊數字的顏色都是單色因此我們轉成28*28*1的資料維度當作未來CNN Model的input
x_train4d = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_test4d = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')

# print(x_train4d.shape)

#灰階的圖片數值為0~255之間，我們將它縮放到0~1之間
#將數值縮小到0~1
x_train4d_normalize = x_train4d / 255
x_test4d_normalize = x_test4d / 255

#對類別資料做onehot-encoding處理
y_trainOneHot = np_utils.to_categorical(y_train)
y_testOneHot = np_utils.to_categorical(y_test)

# print(y_train)
# print(y_trainOneHot)

#建立CNN模型
model = Sequential()

# filter為16, kernel size為(5,5), padding(填充)為(same)
model.add(Conv2D(filters = 16,
                 kernel_size = (5,5),
                 padding = 'same',
                 input_shape = (28,28,1),
                 activation = 'relu'))

#MaxPooling size為(2,2)
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(filters = 36,
                 kernel_size = (5,5),
                 padding = 'same',
                 activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

# Drop掉部分神經元避免overfitting，在層跟層之間通常會Drop掉一定比例的神經元來避免Overfit的狀況，要Drop掉多少比例沒有一個特定的值，通常是25%~50%之間
model.add(Dropout(0.25))

#平坦化
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))

model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

# print(model.summary())

# 訓練模型

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
train_history = model.fit(x = x_train4d_normalize,
                          y = y_trainOneHot,validation_split = 0.2,epochs = 20, batch_size = 300,verbose = 2)

def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


show_train_history('acc','val_acc')
show_train_history('loss','val_loss')
