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


def reshapeImage(directory):
    listPerson = os.listdir(directory)
    for person in listPerson:
        folder = directory + '/' + person
        listImage = os.listdir(folder)
        for image in listImage:
            path = folder + '/' + image
            face = cv2.imread(path)
            face = cv2.resize(face, (160, 160))
            cv2.imwrite(path, face)


class detectFaceData:
    def __init__(self):
        self.target_size = (160, 160)
        self.detector = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True)

    def extractFace(self, dire):
        listDir = os.listdir(dire)
        for nameFolder in listDir:
            folder = dire + '/' + nameFolder
            listImage = os.listdir(folder)
            print(folder)
            for imagePath in listImage:
                path = folder + '/' + imagePath
                print(path)
                image = cv2.imread(path)
                boxes, _ = self.detector.detect(image)
                if boxes is not None:
                    for box in boxes:
                        bbox = list(map(int, box.tolist()))
                        image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        if image is not None:
                            cv2.imwrite(path, image)
                else:
                    os.remove(path)


if __name__ == "__main__":
    reshapeImage(directory)
