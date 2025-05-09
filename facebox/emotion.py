import cv2
import numpy as np

emotion_net = cv2.dnn.readNetFromONNX("./emotion-ferplus-8.onnx")
emotion_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
emotion_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

emotions = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

def detect_emotion(frame, face):
    (x, y, w, h) = face
    face_img = frame[y:y+h, x:x+w]
    face_blob = cv2.dnn.blobFromImage(frame, 1.0, (64, 64), [127,127,127], False, crop=True)
    emotion_net.setInput(face_blob)
    prob = emotion_net.forward()
    emotion_index = np.argmax(prob)
    return emotions[emotion_index]