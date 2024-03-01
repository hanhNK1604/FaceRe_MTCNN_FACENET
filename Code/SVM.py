import cv2
import os
from facenet_pytorch import MTCNN
import torch
import numpy as np
import matplotlib.pyplot as plt
from keras_facenet import FaceNet

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import joblib

data = np.load(r'/home/hanhdz/deepLearning/ProJect/FaceRecog_FaceNet/Code/faces_embeddings.npz')
Xfeature, label = data['arr_0'], data['arr_1']
print(label)

listPerSon ={
    'HoangAnh': 0,
    'BangVu': 1,
    'MinhHieu': 2
}

new_label = []
for i in label:
    new_label.append(listPerSon[i])
print(new_label)

X_train, X_test, Y_train, Y_test = train_test_split(Xfeature, new_label, shuffle=True, train_size=0.8)
model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)
ypreds_train = model.predict(X_train)
ypreds_test = model.predict(X_test)

print(accuracy_score(Y_train, ypreds_train))
file = 'finalModel.joblib'
joblib.dump(model, file)
