import cv2
import os
from facenet_pytorch import MTCNN
import torch
import numpy as np
import matplotlib.pyplot as plt
from keras_facenet import FaceNet

directory = r'/home/hanhdz/deepLearning/ProJect/FaceRecog_FaceNet/DataImage'
Xfeature = []
label = []
EMBEDDED_X = []
embedder = FaceNet()

def get_embedding(face_img):
    face_img = face_img.astype('float32') # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0)
    yhat= embedder.embeddings(face_img)
    return yhat[0] # 512D image (1x1x512)

def getFeature_label(directory):
    listPerson = os.listdir(directory)
    for person in listPerson:
        folder = directory + '/' + person
        listImage = os.listdir(folder)
        for image in listImage:
            path = folder + '/' + image
            face = cv2.imread(path)
            Xfeature.append(face)
            label.append(person)


if __name__ == "__main__":
    getFeature_label(directory)
    for img in Xfeature:
        EMBEDDED_X.append(get_embedding(img))
    EMBEDDED_X = np.asarray(EMBEDDED_X)
    np.savez_compressed('faces_embeddings.npz', EMBEDDED_X, label)
    print(EMBEDDED_X)
    print(label)