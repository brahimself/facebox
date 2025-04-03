import cv2

def load_detector(xml_path: str = "haarcascade_frontalface_default.xml") -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier()
    if not detector.load(cv2.samples.findFile(xml_path)):
        raise IOError("Error loading face cascade")
    return detector

def detect_and_display(detector: cv2.CascadeClassifier, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv2.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
    return frame
