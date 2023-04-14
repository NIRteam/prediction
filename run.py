import cv2
import argparse
import os
# import numpy as np
from common.logging import configure_logger
from metrics.metrics import cosine_similarity_metric, hamming_distance_metric,\
    mse_metric, psnr_metric, ssim_metric

logger = configure_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple model evaluation pipeline")
    parser.add_argument(
        '--first', type=str, default='data/images/pred1.png',
        help='path to input image1')
    parser.add_argument(
        '--second', type=str, default='data/images/real1.png',
        help='path to input image2')
    args = parser.parse_args()

    return args


def model(input):
    return input


def run_metrics(img1, img2):
    mse_res = mse_metric(img1, img2)
    logger.info(f'MSE: {mse_res}')
    ssim_res = ssim_metric(img1, img2)
    logger.info(f'SSIM (not verified): {ssim_res}')
    psnr_res = psnr_metric(img1, img2)
    logger.info(f'PSNR : {psnr_res}')
    hdm_res = hamming_distance_metric(img1, img2)
    logger.info(f'HDM: {hdm_res}')
    csm_res = cosine_similarity_metric(img1, img2)
    logger.info(f'CSM: {csm_res}')


def main(args):

    img1 = cv2.imread(args.first)
    img2 = cv2.imread(args.second)

    if img1.shape != img2.shape:
        logger.warning(
            'Image shapes are different! Scaling img2 to img1 shape...')
        img2 = cv2.resize(
            img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

    if not os.path.exists('data/out'):
        os.makedirs('data/out', exist_ok=True)

    run_metrics(img1, img2)


if __name__ == '__main__':
    logger.debug('Running...')

    args = parse_args()
    main(args)
