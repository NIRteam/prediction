import sys
from common.logging import configure_logger
from common.utils import check_args, create_data_dirs, get_imgs, parse_args

import json

import cv2

from metrics import predictVideo, collectMetrcisSubsequent


def write_frames_and_metrics(name, real, fake, frame_metrics):
    name.mkdir()
    
    cv2.imwrite(str(name / "real.png"), real)
    cv2.imwrite(str(name / "fake.png"), fake)

    with open(name / "metrics.txt", 'w') as f:
        f.write(frame_metrics)


def getModelByName(model_name):
    if model_name.lower() == "dmvfn":
        model = Model(
            DMVFN(load_path = "./pretrained_models/dmvfn_city.pkl"))
    elif model_name.lower() == "vpvi":
       model = Model(
            VPvI(model_load_path = "./pretrained_models/flownet.pkl",
                 flownet_load_path = "./pretrained_models/raft-kitti.pth"))
    else:
        raise ValueError(f"Model {model_name} does not exist")


def subsequentPrediction(*args, **kwargs):
    return collectMetrcisSubsequent(*args, **kwargs)


def patternPrediction(*args, **kwargs):
    iterator = predictVideoIterator(*args, **kwargs)

    for i, (real, fake, frame_metrics, metrics) in enumerate(iterator):
        write_frames_and_metrics(out_path / (str(i) + "_run"), real, fake, frame_metrics)

    return metrics


def main(args):

    input_path = Path(args.input)
    
    out_path_parent = Path("output")
    out_path_parent.mkdir(exist_ok=True)

    subsequent = args.subsequent
    
    # for both predictions
    w, h = args.w, args.h
    model_name = args.model
    
    # for subsequent prediction
    frames2predict = args.frames2predict

    # regular prediction
    # save_path = out_path
    pattern = None # args.pattern
    real = args.real
    fake = args.fake

    model = getModelByName(model_name)

    logger.info(f"Using {args.model.lower} model")

    metrics_json = {}

    for name in os.listdir(input_path):
        logger.info(f"processing file {name}")

        full_path = input_path / name

        name = name.replace(".mp4", "")

        out_path = out_path_parent / (name + "_run")

        if subsequent:
            metrics = subsequentPrediction(model, full_path, frames2predict=frames2predict)

            metrics_json["name"] = metrics
        else:
            video_save_path = out_path / (name + ".mp4")

            metrics = patternPrediction(
                model, full_path, video_save_path, w=w, h=h,
                pattern=pattern, real=real, fake=fake)

            metrics_json["name"] = metrics
    
    with open(out_path_parent / "metrics.json", 'w') as f:
        json.dump(metrics_json, f)


if __name__ == '__main__':

    logger = configure_logger(__name__)

    logger.info('Starting...')

    args = parse_args()

    logger.info(args)

    main(args, logger)
