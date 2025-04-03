import cv2
from facebox.detector import load_detector, detect_and_display

def main():
    detector = load_detector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video stream")
        return

    while True:
        ret, frame = cap.read()
        if frame is None:
            print("No captured frame. Exiting.")
            break

        frame = detect_and_display(detector, frame)
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(10) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
