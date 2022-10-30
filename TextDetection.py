import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D

import pickle

path = 'Resources/myData/Numbers'
testRatio = 0.2
validationRatio = 0.2
imageDimension = (32,32,3)

myList = os.listdir(path)
print("Total no of classes detected: ",len(myList))
noOfClasses = len(myList)
images = []
classNo = []
classDigit = []
noOfSamples=[]
incount = 0
print("Importing Images...")
for folder in myList:
    count = 0
    new_path = os.listdir(path + '/' + folder)
    for image in new_path:
        curImg = cv2.imread(path + '/' + folder + '/' + image)
        curImg = cv2.resize(curImg,(imageDimension[0],imageDimension[1]))
        images.append(curImg)
        classNo.append(incount)
        count+=1
    print(folder,end=" ")
    classDigit.append(folder)
    incount+=1
    noOfSamples.append(count)
print(" ")
print(f"Classes: {classDigit}")
print(str(len(images)) + " images successfully imported.")
images = np.array(images)
classNo = np.array(classNo)

print("Data Shapes: ")
print(images.shape)
print(classNo.shape)

#Data split

X_train,X_test,Y_train,Y_test = train_test_split(images,classNo,test_size=testRatio)
X_train,X_validation,Y_train,Y_validation = train_test_split(X_train,Y_train,test_size=validationRatio)

print(f"\nTraining shape: {X_train.shape}\nTesting shape: {X_test.shape}")
print(f"Validation shape: {X_validation.shape}")

plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses),noOfSamples)
plt.title("No of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

X_train = np.array(list(map(preProcessing,X_train)))
X_test = np.array(list(map(preProcessing,X_test)))
X_validation = np.array(list(map(preProcessing,X_validation)))

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)


#Data Augmentation
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
Y_train = to_categorical(Y_train,noOfClasses)
Y_test= to_categorical(Y_test,noOfClasses)
Y_validation = to_categorical(Y_validation,noOfClasses)

def Model():
    noOfFilters = 60
    sizeOfFilter1 =(5,5)
    sizeOfFilter2 =(3,3)
    sizeOfPool=(2,2)
    noOfNode=500

    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape=(imageDimension[0],imageDimension[1],1),activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation='softmax'))

    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model = Model()
print(model.summary())

batch_size_val = 20
epoch_val=10
steps_per_epoch_val=650

history = model.fit_generator(dataGen.flow(X_train,Y_train,batch_size=batch_size_val),steps_per_epoch=steps_per_epoch_val,epochs=epoch_val,validation_data=(X_validation,Y_validation),shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training','Validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')

plt.show()

score = model.evaluate(X_test,Y_test,verbose=0)
print(f'Test Score = {score[0]}\nTest Accuracy = {score[1]}')

print("Saving Model...")
pickle_out=open("Models/saved_Model.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()
print("Model Saved.")