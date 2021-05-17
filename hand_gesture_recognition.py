import numpy 
import os
from PIL import Image 

dictionary = dict()
count = 0
for j in os.listdir('../dataset/00/'):             
    dictionary[j] = count
    count = count + 1
dictionary


x_data = []
y_data = []
datasize = 0 
for i in range(0, 10): 
    for j in os.listdir('../dataset/0' + str(i) + '/'):  
        count = 0 
        for k in os.listdir('../dataset/0' + str(i) + '/' + j + '/'):     
            img = Image.open('../dataset/0' + str(i) + '/' + j + '/' + k).convert('L')
            img = img.resize((320, 120))
            arr = numpy.array(img)
            x_data.append(arr) 
            count = count + 1
        y_values = numpy.full((count, 1), dictionary[j]) 
        y_data.append(y_values)
        datasize = datasize + count
x_data = numpy.array(x_data, dtype = 'float32')
y_data = numpy.array(y_data)
y_data = y_data.reshape(datasize, 1) 



    
import keras
from keras.utils import to_categorical

y_data = to_categorical(y_data)


x_data = x_data.reshape((datasize, 120, 320, 1))
x_data /= 255


from sklearn.model_selection import train_test_split
x_train,x_remaining,y_train,y_remaining = train_test_split(x_data,y_data,test_size = 0.2)
x_validate,x_test,y_validate,y_test = train_test_split(x_remaining,y_remaining,test_size = 0.5)

alexnet = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(120, 320,1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])



alexnet.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

alexnet.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(x_validate, y_validate))


