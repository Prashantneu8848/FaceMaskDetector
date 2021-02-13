import tensorflow as tf
import cv2
import numpy as np


img = cv2.imread("./Images/Dr.Feynman.jpg")
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
