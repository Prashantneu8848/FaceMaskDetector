import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from create_face_roi import detect_face_in_image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tkinter import filedialog
# from video import detect_face_in_video

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims

def fetch_image_and_predict(img_path):
    img_array_expanded_dims = prepare_image(img_path)
    predictions = mask_classify_v1.predict(tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims))
    result = 'wearing mask' if predictions[0][0] > predictions[0][1] else 'not wearing mask'
    return result

# mask_classify_v1 = load_model('models/mask_classify_v1.h5')

root = tk.Tk()
info_frame = tk.Frame(master=root, relief=tk.RIDGE, borderwidth=5)

button_frame = tk.Frame(master=root, relief=tk.RIDGE, borderwidth=5)

info = tk.Label(
    master = info_frame,
    text = 'Face Mask Detector. Beta Release',
    foreground = 'white',
    background = 'black',
)
info.pack(side=tk.TOP)

btn_image = tk.Button(
    master = button_frame,
    text = 'Classify Images',
    width = 25,
    height = 5,
    highlightbackground='#3E4149',
    foreground = 'black',
    background = 'white'
)
btn_image.pack(side=tk.LEFT)

btn_video = tk.Button(
    master = button_frame,
    text = 'Classify Video',
    width = 25,
    height = 5,
    highlightbackground='#3E4149',
    foreground = 'black',
    background = 'white'
)
btn_video.pack(side=tk.RIGHT)

info_frame.pack()
button_frame.pack()

root.mainloop()
# root.withdraw()

# img_path = filedialog.askopenfilename()

# if img_path:
#     result = fetch_image_and_predict(img_path)
#     img = cv2.imread(img_path)
#     detect_face_in_image(img, result)
# else:
#     print("user did not select any images.")

# detect_face_in_video()