import tensorflow as tf
import cv2
import numpy as np
import imutils
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

face_classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
eyes_classifier = cv2.CascadeClassifier("./haarcascade_frontal_eyes.xml")
mask_classify_v1 = load_model('models/mask_classify_v1.h5')

def prepare_frame(img):
    resize = cv2.resize(img, (224, 224))
    img_iv_color = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_iv_color)
    img_array = image.img_to_array(im_pil)
    
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims


def fetch_image_and_predict(img):
    img_array_expanded_dims = prepare_frame(img)
    predictions = mask_classify_v1.predict(tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims))
    result = 'wearing mask' if predictions[0][0] > predictions[0][1] else 'not wearing mask'
    return result

"""
Displays live video with face ROI
"""
def detect_face_in_video():
    #Use the default camera.
    video = cv2.VideoCapture(0)
    # video.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        frame = imutils.resize(frame, width = 400)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        result = fetch_image_and_predict(frame)

        font = cv2.FONT_HERSHEY_SIMPLEX

        face = face_classifier.detectMultiScale(gray_frame, 1.1, 4)
        eyes = eyes_classifier.detectMultiScale(gray_frame, 1.1, 4)

        if len(face) != 0:
            (x, y, w, h) = face[0]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(frame, result, (x,y), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        if len(eyes) != 0:
            (x_e, y_e, w_e, h_e) = eyes[0]

            cv2.rectangle(frame, (x_e, y_e), (x_e + w_e, y_e + h_e), (0, 255, 0), 2)
        
        cv2.imshow('Face Detection', frame)
    
        k = cv2.waitKey(1)

        #Closes window when ESC button is pressed.
        # Modify k = cv2.waitKey(0) & 0xFF for 64-bit machine
        if k == 27:
            cv2.destroyAllWindows()
            break

    video.release()

detect_face_in_video()