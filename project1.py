import cv2
import os

name = input("Enter Name: ")

path = f"dataset/{name}"
os.makedirs(path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

count = 0

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
        1.1,
        5
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        count += 1

        cv2.imwrite(
            f"{path}/{count}.jpg",
            face
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

    cv2.imshow(
        "Dataset Collection",
        frame
    )

    if count >= 50:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Dataset Collection Complete")

