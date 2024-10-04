import threading
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 440)

counter = 0
face_match = False
lock = threading.Lock()
reference_img = cv2.imread("reference.jpg")

def check_face(frame):
    global face_match
    try:
        with lock:
            result = DeepFace.verify(frame, reference_img.copy())
            face_match = result['verified']
            print(f"Verification result: {face_match}")  # Debug output
    except ValueError as e:
        with lock:
            face_match = False
            print(f"Error: {e}")  # Debug output

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            threading.Thread(target=check_face, args=(frame.copy(), )).start()
        counter += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_roi = frame[y:y+h, x:x+w]

        with lock:
            if face_match:
                cv2.putText(frame, "Its Carrey", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "None", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
