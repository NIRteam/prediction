import sys
from common.logging import configure_logger
from common.utils import check_args, create_data_dirs, get_imgs, parse_args, read_img_from_path, scale_imgs, write_result, write_result_img
from metrics.metrics import run_metrics
from model import Model, DMVFN, VPvI
import cv2
logger = configure_logger(__name__)


def main(args):

    exp_dir = create_data_dirs(args)

    # if not check_args(args):
    #     sys.exit('Incorrect args!\n Exiting...')

    # images = get_imgs(args)
    images_paths = ['data/input/FPV/1.png', 'data/input/FPV/2.png']

    images = read_img_from_path(images_paths)
    # for image_path in images_paths:
    #     cv2.imread(img1_path)

    images = scale_imgs(images, (640, 640))

    model = Model(DMVFN(load_path = "./pretrained_models/dmvfn_city.pkl"))
    
    logger.debug(len(images))    
    result_image = model.predict(images, num_frames_to_predict=1)
    cv2.imwrite('test_00.jpg', result_image)
    write_result_img(result_image)
    logger.debug(len(result_image))    

    # write_result_img(result_image)
    # logger.debug(result_image)

    # for index, image_pair in enumerate(images):
    #     run_metrics(*image_pair, exp_dir, index)


if __name__ == '__main__':
    logger.info('Starting...')

    args = parse_args()
    main(args)
