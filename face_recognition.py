import cv2
import numpy
import os

haar_file = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)
datasets = 'datasets'

print("Training...")

(images, labels, names, id) = ([], [], {}, 0)
camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(grayImg, 1.3, 4) # get coordinates of face
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = grayImg[y:y+h, x:x+w]
        
    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(10) & 0xFF
     
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows() 