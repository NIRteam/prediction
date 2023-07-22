import cv2
import argparse
import os
import shutil
import sys

from pathlib import Path

from common.logging import configure_logger
logger = configure_logger(__name__)


def getFrames(
    video_path : str,
    files : list = None,
    w = 1280,
    h = 720):

    video_path = Path(video_path)

    if not video_path.exists():
        raise ValueError(f"Path f{video_path} does not exist")

    if video_path.suffix in [".mp4",]: ### TODO add other extensions
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for i in tqdm(range(frame_count)):
            ret, frame = cap.read()

            if not ret:
                cap.release()
                raise RuntimeError(f"Could not read video f{video_path}")

            yield cv2.resize(frame, (w, h))

        cap.release()


    else:
        if not video_path.is_dir():
            raise ValueError(f"Path f{video_path} is not a directory nor a video file")

        files = os.listdir(video_path)
        
        for i in tqdm(range(len(files))):
            yield read_img(video_path / files[i], w, h)


def read_img(path, w=1280, h=720):
    # img = Image.open(path)
    # img = np.array(img)
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))
    return img


def create_data_dirs(args):
    """По этому поводу хотел обсудить со всеми:
    1.Проверять папку output смысла нет, ее можно создать при первом запуске
    2.Есть вариант что пользователь хочет закинуть какой-то конкретный инпут в 
    прогу (пока у нас нет настоящего инпута). У нас есть 2 варианта как это 
    обеспечить 
    сделать ему интерфейс который заставит его перемещать файлы в папку, 
    либо вводить инпут аргументом, имхо аргумент логичнее, в таком случае 
    наличие 
    папки input не обязательно"""
    if args.clean:
        shutil.rmtree('data/output')
    os.makedirs('data/output', exist_ok=True)
    if not os.path.exists('data/input'):
        os.makedirs('data/input', exist_ok=True)
        sys.exit('Необходимые директории отсутствуют!\n'
                 'Создана директория data/input')
    exp_dir = create_exp_dir()
    return exp_dir


# def get_imgs_from_img(img1_path, img2_path):
#     img1 = cv2.imread(img1_path)
#     img2 = cv2.imread(img2_path)

#     if img1.shape != img2.shape:
#         logger.warning(
#             'Image shapes are different! Scaling img2 to img1 shape...')
#         img2 = cv2.resize(
#             img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

#     return img1, img2


def get_imgs_from_file(file_path):
    try:
        file = open(file_path)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(f'Input file not found: {file_path}')
    else:
        img_list = []
        with file:
            for line in file.readlines():
                image_paths = line.split(',')
                image_paths = [path.strip() for path in image_paths]
                img1_path, img2_path = image_paths
                img_list.append(get_imgs_from_img(img1_path, img2_path))
        return img_list


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple model evaluation pipeline")
    
    parser.add_argument(
        '--input', type=str, default="input",
        help="path to input dir")

    parser.add_argument(
        '--model_name', type=str, default="DMVFN",
        help="which model to use for prediction. DMVFN or VPvI. DMVFN by default")

    parser.add_argument(
        '--w', type=int, default=1280,
        help="width of procesed video")

    parser.add_argument(
        '--h', type=int, default=720,
        help="height of procesed video")

    ### TODO extend description
    parser.add_argument(
        '--subsequent', type=bool, default=False,
        help="subsequent prediction. Normal video prediction by default")

    parser.add_argument(
        '--frames2predict', type=int, default=5,
        help="how many frames to predict in subsequent predcition")

    ### TODO extend description
    parser.add_argument(
        '--real', type=int, default=2,
        help="how many real frames in predcition pattern for regular predcition")

    parser.add_argument(
        '--fake', type=int, default=1,
        help="how many fake frames in predcition pattern for regular predcition")
    
    # parser.add_argument(
    #     '--clean', action='store_true',
    #     help='add this arg to remove all previous experiments from output')
    
    args = parser.parse_args()

    return args


# def create_out_dir():
#     max_id = 0
#     for dir in os.listdir('data/output'):
#         cur_id = int(dir.split('_')[1])
#         if cur_id > max_id:
#             max_id = cur_id

#     new_exp_dir = f'data/output/exp_{max_id + 1}'
#     os.makedirs(new_exp_dir)
#     return new_exp_dir


# def write_result(msg: str, logger, out_file):
#     logger.info(msg)
#     out_file.write(f'{msg}\n')


# def check_args(args):
#     if not (args.file or args.first or args.second):
#         logger.error('no args')
#         return False

#     if args.file and (args.first or args.second):
#         logger.error('Only one type of input ins supported, file or images')
#         return False

#     if (args.first and not args.second) or (args.second and not args.first):
#         logger.error('input two images --first and --second or use txt file')
#         return False

#     return True


def scale_images(imgs_list):
    return imgs_list
