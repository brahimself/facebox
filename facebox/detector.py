import cv2
import numpy as np

net = cv2.dnn.readNetFromTorch('./openface.nn4.small2.v1.t7')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_frontalface_default.xml')

def load_signatures(image_paths, label):
    signature_total = np.zeros(128)
    for image in image_paths:
        frame = cv2.imread(image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locations = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in face_locations:
            face = frame[y:y+h, x:x+w]
            blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            net.setInput(blob)
            signature_total += net.forward().flatten()
    return (signature_total / len(image_paths)).flatten()


def get_signature(face):
    blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    return net.forward()

