from flask import Flask, request, send_file, abort
import cv2
import numpy as np
import requests
import tempfile
import os

app = Flask(__name__)

def download_image(url):
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except:
        return None

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    return sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]

def crop_centered(img, x, y, w, h, crop_width=600, crop_height=800):
    face_center_x = x + w // 2
    face_center_y = y + h // 2

    start_x = max(0, face_center_x - crop_width // 2)
    start_y = max(0, face_center_y - crop_height // 2)

    end_x = min(start_x + crop_width, img.shape[1])
    end_y = min(start_y + crop_height, img.shape[0])
    crop = img[start_y:end_y, start_x:end_x]

    return cv2.resize(crop, (crop_width, crop_height))

def process_image(url):
    img = download_image(url)
    if img is None:
        return None
    face = detect_face(img)
    if face is None:
        return None
    x, y, w, h = face
    return crop_centered(img, x, y, w, h)

@app.route('/merge', methods=['GET'])
def merge_faces():
    url1 = request.args.get('url1')
    url2 = request.args.get('url2')
    if not url1 or not url2:
        return "Both url1 and url2 parameters are required", 400

    img1 = process_image(url1)
    img2 = process_image(url2)

    if img1 is None or img2 is None:
        return "One or both faces could not be processed", 400

    separator = np.full((800, 3, 3), 255, dtype=np.uint8)  # 3px genişliğinde beyaz şerit
    combined = np.hstack((img1, separator, img2))
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_file.name, combined)
    return send_file(temp_file.name, mimetype='image/jpeg')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
