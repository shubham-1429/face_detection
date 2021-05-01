#For image
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('C:/Users/dell/Desktop/face_detection/dataset/test.jpg')
#paste the dataset path here 

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),1)

cv2.imshow('img',img)
cv2.waitKey()
'''
#For live camera or video 
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0) #if you want camera 
#cap = cv2.VideoCapture('dataset/test2.mp4') #to get faces from video then use ('datafilename.mp4')

while True:
    _, img = cap.read()
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,0,255),1)#(0,0,255) for red ()

    cv2.imshow('img',img)

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()    
'''