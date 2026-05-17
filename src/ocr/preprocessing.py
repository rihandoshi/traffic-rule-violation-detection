import cv2
import numpy as np
import os

from streamlit import image

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from path: {image_path}")  
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img

def deskew(image):
    co_ords = np.column_stack(np.where(image > 0))
    if(len(co_ords) == 0):
        return image
    angle = cv2.minAreaRect(co_ords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def scale_image(image):
    scaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(scaled, -1, kernel)
    return sharpened

def remove_noise(image):
    denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    return denoised

def thinning_and_skeletonization(image):
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=1)
    skeleton = cv2.ximgproc.thinning(erosion)
    return skeleton

def get_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def thresholding(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def opening(image):
    kernel = np.ones((3,3), np.uint8)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened

def closing(image):
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed

