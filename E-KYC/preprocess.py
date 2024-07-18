import cv2
import numpy as np
import os
import logging
from utils import read_yaml, file_exists

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "ekyc_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

config_path = "config.yaml"
config = read_yaml(config_path)

artifacts = config['artifacts']
intermediate_dir_path = artifacts['INTERMIDEIATE_DIR']
contour_file_name = artifacts['CONTOUR_FILE']

def read_image(image_path, is_uploaded=False):
    if is_uploaded:
        try:
            if image_path is None:
                logging.error("Uploaded image file is None.")
                raise ValueError("Uploaded image file is None.")
            # Read image using OpenCV
            image_bytes = image_path.read()
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                logging.error("Failed to decode uploaded image.")
                raise ValueError("Failed to decode uploaded image.")
            return img
        except Exception as e:
            logging.error(f"Error reading uploaded image: {e}")
            print("Error reading uploaded image:", e)
            return None
    else:
        try:
            if not os.path.exists(image_path):
                logging.error(f"Image path does not exist: {image_path}")
                raise ValueError(f"Image path does not exist: {image_path}")
            img = cv2.imread(image_path)
            if img is None:
                logging.error(f"Failed to read image from path: {image_path}")
                raise ValueError(f"Failed to read image from path: {image_path}")
            return img
        except Exception as e:
            logging.error(f"Error reading image from path: {e}")
            print("Error reading image from path:", e)
            return None

def extract_id_card(img):
    try:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = None
        largest_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > largest_area:
                largest_contour = cnt
                largest_area = area

        if largest_contour is None:
            logging.error("No contours found in the image.")
            return None, None

        x, y, w, h = cv2.boundingRect(largest_contour)
        logging.info(f"Contours are found at: {(x, y, w, h)}")

        current_wd = os.getcwd()
        filename = os.path.join(current_wd, intermediate_dir_path, contour_file_name)
        contour_id = img[y:y+h, x:x+w]
        is_exists = file_exists(filename)
        if is_exists:
            os.remove(filename)

        cv2.imwrite(filename, contour_id)

        return contour_id, filename
    except Exception as e:
        logging.error(f"Error extracting ID card: {e}")
        print("Error extracting ID card:", e)
        return None, None

def save_image(image, filename, path="."):
    try:
        full_path = os.path.join(path, filename)
        is_exists = file_exists(full_path)
        if is_exists:
            os.remove(full_path)

        cv2.imwrite(full_path, image)

        logging.info(f"Image saved successfully: {full_path}")
        return full_path
    except Exception as e:
        logging.error(f"Error saving image: {e}")
        print("Error saving image:", e)
        return None

def detect_and_extract_face(img):
    try:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(config['artifacts']['HAARCASCADE_PATH'])

        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

        max_area = 0
        largest_face = None
        for (x, y, w, h) in faces:
            area = w * h
            if area > max_area:
                max_area = area
                largest_face = (x, y, w, h)

        if largest_face is not None:
            (x, y, w, h) = largest_face
            new_w = int(w * 1.50)
            new_h = int(h * 1.50)
            new_x = max(0, x - int((new_w - w) / 2))
            new_y = max(0, y - int((new_h - h) / 2))

            extracted_face = img[new_y:new_y+new_h, new_x:new_x+new_w]

            filename = os.path.join(os.getcwd(), config['artifacts']['INTERMIDEIATE_DIR'], "extracted_face.jpg")
            if os.path.exists(filename):
                os.remove(filename)

            cv2.imwrite(filename, extracted_face)
            logging.info(f"Extracted face saved at: {filename}")
            return filename
        else:
            logging.error("No faces detected in the image.")
            return None
    except Exception as e:
        logging.error(f"Error detecting and extracting face: {e}")
        print("Error detecting and extracting face:", e)
        return None
