import cv2
import os
from facenet_pytorch import MTCNN
import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
from keras_facenet import FaceNet

class FinalModel:
    def __init__(self):
        self.detector = MTCNN(thresholds=[0.6, 0.6, 0.8], keep_all=True)
        self.target_size = (160, 160)
        self.embedder = FaceNet()
        self.linkToModel = r'/home/hanhdz/deepLearning/ProJect/FaceRecog_FaceNet/Code/finalModel.joblib'
        self.pretrainModel = joblib.load(self.linkToModel)

    def getImage(self, frame):
        listImage = []
        boxes, _ = self.detector.detect(frame)
        if boxes is not None:
            for box in boxes:
                bbox = list(map(int, box.tolist()))
                image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                listImage.append(image)
        return listImage

    def resizeImage(self, listImage):
        resizeImage = []
        for image in listImage:
            image = cv2.resize(image, self.target_size)
            resizeImage.append(image)
        return resizeImage

    def convertImage(self, resizeImage):
        embeddedX = []
        for image in resizeImage:
            image = image.astype('float32')
            image = np.expand_dims(image, axis=0)
            yhat = self.embedder.embeddings(image)
            embeddedX.append(yhat)

        embeddedX = np.asarray(embeddedX)
        return embeddedX

    def getPrediction(self, embeddedX):
        predictions = []
        listPerSon = {
            0: 'HoangAnh',
            1: 'BangVu',
            2: 'MinhHieu'
        }
        for feature in embeddedX:
            prediction = self.pretrainModel.predict(feature)
            prediction = list(prediction)
            predictions += prediction

        predictions = list(predictions)
        predictions = [listPerSon[i] for i in predictions]
        return predictions

    def showWinPred(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while cap.isOpened():
            isSuccess, frame = cap.read()
            if isSuccess:
                boxes, _ = self.detector.detect(frame)
                listImage = self.getImage(frame)
                if len(listImage) != 0:
                    resizeImage = self.resizeImage(listImage)
                    embeddedX = self.convertImage(resizeImage)
                    predictions = self.getPrediction(embeddedX)
                    for index, box in enumerate(boxes):
                        bbox = list(map(int, box.tolist()))
                        frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
                        frame = cv2.putText(frame, predictions[index], (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.imshow('FACE RECOGNIZATION', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    finalModel = FinalModel()
    finalModel.showWinPred()












