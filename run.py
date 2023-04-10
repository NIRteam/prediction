import cv2
import argparse
import os
import numpy as np
from common.logging import configure_logger

logger = configure_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple model evaluation pipeline")
    parser.add_argument(
        '-i', '--input', type=str, default='data/images/img.jpg',
        help='path to input file')
    args = parser.parse_args()

    return args


def model(input):
    return input


def compare(img1, img2):
    diff = cv2.absdiff(img1, img2)
    res = diff.astype(np.uint8)
    # cv2.imwrite('data/out/out.jpg', res)
    percentage = np.count_nonzero(res) / res.size
    logger.info('Results:\n'
                f' comparison: {percentage}')


def main(args):

    input = cv2.imread(args.input)
    result = model(input)

    if not os.path.exists('data/out'):
        os.makedirs('data/out', exist_ok=True)

    cv2.imwrite('data/out/out.jpg', result)

    compare(input, result)


if __name__ == '__main__':
    logger.debug('Running...')

    args = parse_args()
    main(args)
