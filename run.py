import sys
import os
import shutil
import json
import cv2

from tqdm.auto import tqdm

from pathlib import Path

from common.logging import configure_logger
from common.utils import parse_args

from metrics import predictVideoIterator, collectMetrcisSubsequent

from model import Model, DMVFN, VPvI


def write_frames_and_metrics(name, real, fake, frame_metrics):
    name.mkdir()
    
    cv2.imwrite(str(name / "real.png"), real)
    cv2.imwrite(str(name / "fake.png"), fake)

    frame_metrics = str(frame_metrics)

    frame_metrics = frame_metrics.replace('{', "").replace('}', "")
    frame_metrics = frame_metrics.replace("'", "")
    frame_metrics = frame_metrics.replace(':', '=')

    with open(name / "metrics.txt", 'w') as f:
        f.write(frame_metrics)


def getModelByName(model_name, device):
    if model_name.lower() == "dmvfn":
        model = Model(
            DMVFN(load_path = "model/pretrained_models/dmvfn_city.pkl", device=device))
    elif model_name.lower() == "vpvi":
       model = Model(
            VPvI(model_load_path = "model/pretrained_models/flownet.pkl",
                 flownet_load_path = "model/pretrained_models/raft-kitti.pth",
                 device = device,))
    else:
        raise ValueError(f"Model {model_name} does not exist")

    return model


def subsequentPrediction(*args, **kwargs):
    return collectMetrcisSubsequent(*args, **kwargs)


def patternPrediction(out_path, *args, **kwargs):
    iterator = predictVideoIterator(*args, **kwargs)

    for i, (real, fake, frame_metrics, metrics) in enumerate(iterator):
        write_frames_and_metrics(out_path / (str(i) + "_run"), real, fake, frame_metrics)

    return metrics


def main(args, logger):

    input_path = Path(args.input)

    out_path_parent = Path("output")

    # if os.path.exists(out_path_parent):
    #     shutil.rmtree(out_path_parent)

    out_path_parent.mkdir(exist_ok=True)

    subsequent = args.subsequent

    device = args.device
    
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

    model = getModelByName(model_name, device)

    logger.info(f"Using {model_name.lower()} model")

    metrics_json = {}

    files = os.listdir(input_path)

    for i, name in enumerate(files):
        logger.info(f"processing file {name} {i}/{len(files)}")

        full_path = input_path / name

        name = name.replace(".mp4", "")

        out_path = out_path_parent / (name + "_run")

        if os.path.exists(out_path):
            shutil.rmtree(out_path)

        out_path.mkdir()

        if subsequent:
            metrics = subsequentPrediction(model, full_path, w=w, h=h, frames2predict=frames2predict)

            metrics_json["name"] = metrics
        else:
            video_save_path = out_path / (name + ".mp4")

            print(full_path)

            metrics = patternPrediction(
                out_path,
                model, full_path, video_save_path, w=w, h=h,
                real_fake_pattern=pattern, real=real, fake=fake)

            metrics_json["name"] = metrics
    
    with open(out_path_parent / "metrics.json", 'w') as f:
        json.dump(metrics_json, f)


if __name__ == '__main__':

    logger = configure_logger(__name__)

    logger.info('Starting...')

    args = parse_args()

    logger.info(args)

    main(args, logger)
