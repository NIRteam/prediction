import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity
import cv2
import math


def cosine_similarity_metric(image1, image2):

    image1_vector = np.array(image1).ravel()
    image2_vector = np.array(image2).ravel()

    similarity_score = cosine_similarity(
        [image1_vector], [image2_vector])[0][0]

    return similarity_score


def hamming_distance_metric(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    image1_binary = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)[1]
    image2_binary = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)[1]

    difference = cv2.bitwise_xor(image1_binary, image2_binary)
    hamming_distance = cv2.countNonZero(difference)

    return hamming_distance


def mse_metric(image1, image2):
    mse = np.mean((image1 - image2) ** 2)

    return mse


def psnr_metric(image1, image2):
    mse = mse_metric(image1, image2)
    if mse == 0:
        return 100

    psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    return psnr


def ssim_metric(image1, image2):
    """Needs checking! Probably wrong!"""
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(image1, image2, full=True)

    return score
