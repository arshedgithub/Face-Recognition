import cv2
import os

haar_file = "haarcascade_frontalface_default.xml"
datasets = 'datasets'

while True:
    sub_data = input("Enter name for the dataset: ")
    path = os.path.join(datasets, sub_data)
    
    if os.path.isdir(path):
        print(f"Error: Dataset '{sub_data}' already exists. Please choose a different name.")
    else:
        os.mkdir(path)
        break
    
(width, height) = (130, 100)
    
haar_cascade = cv2.CascadeClassifier(haar_file)
camera = cv2.VideoCapture(0)

count = 1
while count < 31:
    print(f"Image {count} added for {sub_data}")
    _, frame = camera.read()
    
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = haar_cascade.detectMultiScale(grayImg, 1.3, 4) # get coordinates of face
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = grayImg[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s-%s.png' % (path, sub_data, count), face_resize)
        count += 1
        
    cv2.imshow("Face Detection", frame)
    key = cv2.waitKey(10) & 0xFF
     
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows() 