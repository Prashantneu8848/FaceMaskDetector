import tensorflow as tf
import cv2
import numpy as np


img = cv2.imread("./Images/Dr.Feynman.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
face = face_classifier.detectMultiScale(img, 1.3, 5)[0]

x = face[0]
y = face[1]
w = face[2]
h = face[3]

# (x, y, w, h) = face
# x = x - 100
# w = w + 100
# y = y - 100
# h = h + 100

cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
