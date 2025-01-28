import json
import pickle
import numpy as np
import cv2
import wavelet
import base64
from wavelet import w2d
import joblib

__model = None
__class_name_to_number = {}
__class_number_to_name = {}


def classify_image(image_base64_data):
    imgs = get_cropped_image_if_2_eyes(image_base64_data)

    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(
            32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32 * 32 * 3 + 32 * 32  # This is 4096

        final = combined_img.reshape(1, len_image_array).astype(float)
        result.append({
            'class': __class_number_to_name[__model.predict(final)[0]],
            'class_probability': np.around(__model.predict_proba(final)*100, 2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result





def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __model
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/SPC_class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k,
                                  v in __class_name_to_number.items()}

    if __model is None:
        with open("./artifacts/sports_person_classifier.pkl", "rb") as f:
            __model = joblib.load(f)
    print("Loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_cropped_image_if_2_eyes(image_base64_data):
    face_cascade = cv2.CascadeClassifier(
        './opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(
        './opencv/haarcascades/haarcascade_eye.xml')

    img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces
