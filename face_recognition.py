import cv2
import numpy as np
import os

haar_file = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)
datasets = 'datasets'

print("Training...")

(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectPath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectPath):
            path = subjectPath + "/" + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

(images, labels) = [np.array(lis) for lis in (images, labels)]
print(images, labels)

(width, height) = (130, 100)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

print("Training Completed")

camera = cv2.VideoCapture(0)
cnt = 0

while True:
    _, frame = camera.read()
    
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(grayImg, 1.3, 4) # get coordinates of face
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        face = grayImg[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        
        prediction = model.predict(face_resize)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if prediction[1] < 800:
            cv2.putText(frame, f'{names[prediction[0]]} - {prediction[1]:.2f}', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)            
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt += 1
            cv2.putText(frame, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)            
            if cnt > 100:
                print("Unknown Person")
                cv2.imwrite("unknown.jpg", frame)
                cnt = 0
        
    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(10) & 0xFF
     
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows() 