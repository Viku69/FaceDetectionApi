import numpy as np
import pywt
import cv2


def w2d(img, mode='haar', level=1):
    imArray = img
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    imArray = np.float32(imArray) / 255.0  # Convert to float and normalize
    # Compute wavelet coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0  # Set approximation coefficients to 0

    # Reconstruct the image
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)  # Convert back to uint8
    return imArray_H
