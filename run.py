import cv2
import numpy as np
from facebox.emotion import detect_emotion, emotion_net, emotions
from facebox.detector import net, face_cascade, load_signatures, get_signature

elon_images = ["Elon.jpg","Elon2.jpg","Elon3.jpg","Elon4.jpg","Elon5.png","Elon6.png","Elon7.png"]
larry_images = ["Larry1.png","Larry2.png","Larry3.png","Larry4.png","Larry5.png","Larry6.png","Larry7.png"]

signatures = {
    "Elon": load_signatures(elon_images, "Elon"),
    "Larry": load_signatures(larry_images, "Larry")
}

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_locations = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in face_locations:
        face = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        signature = get_signature(face)

        name = "Unknown"
        for key, value in signatures.items():
            distance = np.linalg.norm(value - signature)
            if distance < 0.85:
                name = key
                break

        emotion = detect_emotion(gray, (x, y, w, h))
        emotion_net.setInput(blob)
        scores = net.forward()[0]
        for i, e in enumerate(emotions):
            score = round(scores[i] * 100, 2)
            text = f"{e}: {score}%"
            cv2.putText(frame, text, (10, (i+1)*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, f"{name} | {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Detection de visages', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
