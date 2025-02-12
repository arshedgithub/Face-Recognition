import cv2
import os

haar_file = "haarcascade_frontalface_default.xml"
datasets = 'datasets'
sub_data = 'arshed'

path = os.path.join(datasets, sub_data)
print(path)

haar_cascade = cv2.CascadeClassifier(haar_file)
camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4) # get coordinates of face
    
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    cv2.imshow("Face Detection", frame)
    key = cv2.waitKey(10) & 0xFF
     
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows() 