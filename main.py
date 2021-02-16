import tensorflow as tf
import cv2
import numpy as np


img = cv2.imread("./Images/Dr.Feynman.jpg")
# img = cv2.imread("./Images/Mask.jpg")
face_classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

"""
Loads the given image and finds the image from the face.
"""
def detect_face_in_image(img):
    # If the given image is RGB then convert to gray to reduce complexity.
    if len(img.shape) > 2:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    face = face_classifier.detectMultiScale(gray_img, 1.5, 5)[0]

    (x, y, w, h) = face

    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    display_image(img, "Face Detection")

"""
Displays the image in a window with a message and closes that window after a
key is pressed.
"""
def display_image(img, img_msg):
    
    cv2.imshow(img_msg, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_face_in_image(img)