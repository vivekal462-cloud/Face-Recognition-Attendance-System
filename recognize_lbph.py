import cv2
import csv
import os
from datetime import datetime

# Face Detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Trained Model Load
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Labels
names = {
    0: "vivek"
}
attendance_marked = set()

# Attendance Function
def mark_attendance(name):

    file_name = "attendance.csv"

    # Header create if file doesn't exist
    if not os.path.exists(file_name):
        with open(file_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Date", "Time"])

    today = datetime.now().strftime("%Y-%m-%d")
    already_marked = False

    with open(file_name, "r") as file:
        reader = csv.reader(file)

        for row in reader:
            if len(row) >= 2:
                if row[0] == name and row[1] == today:
                    already_marked = True
                    break

    if not already_marked:
        with open(file_name, "a", newline="") as file:
            writer = csv.writer(file)

            writer.writerow([
                name,
                today,
                datetime.now().strftime("%H:%M:%S")
            ])

        print(f"{name} Attendance Marked")


# Webcam Start
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face_roi = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face_roi)

        if confidence < 100:

            name = names.get(label, "Unknown")
            if name != "Unknown" and name not in attendance_marked:
                mark_attendance(name)
                attendance_marked.add(name)

            accuracy = round(100 - confidence)

            display_text = f"{name} ({accuracy}%)"

        else:

            display_text = "Unknown"

        # Rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        # Name Display
        cv2.putText(
            frame,
            display_text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    cv2.imshow(
        "Face Recognition Attendance System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()