import cv2
from facenet_pytorch import MTCNN
import torch
from finalModel import FinalModel


if __name__ == "__main__":
    finalModel = FinalModel()
    finalModel.showWinPred()
