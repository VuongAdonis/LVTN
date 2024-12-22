import cv2
import numpy as np


def get_limits(color):
    # here insert the bgr values which you want to convert to hsv
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    """
    decrease and increase the hue value with 10
    set the low limit of saturation and value equal 100
    set the upper limit of saturation and value equal 255
    """
    lowerLimit = hsvC[0][0][0] - 10, 100, 100
    upperLimit = hsvC[0][0][0] + 10, 255, 255

    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit
