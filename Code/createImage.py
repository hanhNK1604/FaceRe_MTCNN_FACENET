import cv2
import os
from facenet_pytorch import MTCNN
import torch
import numpy as np
import matplotlib.pyplot as plt

dataVid = r'/home/hanhdz/deepLearning/ProJect/FaceRecog_FaceNet/DataVid'
dataImage = r'/home/hanhdz/deepLearning/ProJect/FaceRecog_FaceNet/DataImage'
listFolderVid = os.listdir(dataVid)

for folderName in listFolderVid:
    dirToVid = dataVid + '/' + folderName + '/' + folderName + '.mp4'
    dirToSaveImage = dataImage + '/' + folderName
    print(dirToSaveImage)
    print(dirToVid)
    cap = cv2.VideoCapture(dirToVid)
    while cap.isOpened():
        isSuccessful, frame = cap.read()
        if isSuccessful:
            print(cap.get(cv2.CAP_PROP_POS_FRAMES))
            pathName = dataImage + '/' + folderName + '/' + folderName + str(int(cap.get(cv2.CAP_PROP_POS_FRAMES))) + '.png'
            cv2.imshow('Video', frame)
            cv2.imwrite(pathName, frame)
        if (cv2.waitKey(1) & 0xFF == 27) | (int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            break
    cap.release()
    cv2.destroyAllWindows()
