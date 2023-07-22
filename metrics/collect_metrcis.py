from ..common.utils import getFrames
from .metrics import compute_metrics, compute_important_metrics, get_metrics_example, get_important_metrics_example

from pathlib import Path


# collect metric for every subsequent frame predicted

# metrics = {
#     frame_predicted : {
#         metric1 : 1,
#         metric2 : 2
#     }
# }

### example of metric

# metrics = {
#     "1.mp4" : {
#          0 : {
#              "mse" : 10,
#              "cosim" : .5,
#              ...
#          },
#          1 : {
#              "mse" : 10,
#              "cosim" : .5,
#              ...
#          },
#          ...
#     }
#
#     "2.mp4" : {
#          0 : {
#              "mse" : 10,
#              "cosim" : .5,
#              ...
#          },
#          1 : {
#              "mse" : 10,
#              "cosim" : .5,
#              ...
#          },
#          ...
#     },
#     ...
# }


def collectMetrcisSubsequent(
    model,
    video_path : str,
    frames_as_input = 2,
    frames2predict = 5,
    w = 1280,
    h = 720):

    frames_iterator = getFrames(video_path, w, h)

    prev = [ # previous frames
        next(frames_iterator) for i in range(frames2predict + frames_as_input - 1)
        ]

    video_metrics = {i : get_metrics_example() for i in range(frames2predict)}

    for new_frame in frames_iterator:
        prev.append(new_frame) # add next frame

        assert len(prev) == (frames2predict + frames_as_input)

        # sometimes there is an error, that i dont know how to replicate or handle
        # it seems to be random
        try:
            fake_frames = model.predict(prev[:2], frames2predict)
        except TypeError:
            # print(traceback.format_exc())
            # continue
            fake_frames = model.predict(prev[:2], frames2predict)

        assert len(fake_frames) == frames2predict

        for j in range(frames2predict):
            # frame_file_name = frames_dir / f"{i+j:04d}.jpg"

            real = prev[2 + j]
            fake = fake_frames[j]

            frame_metrics = compute_metrics(real, fake)

            for metric_name in frame_metrics:
                video_metrics[j][metric_name].append(frame_metrics[metric_name])

        del prev[0] # remove first frame, that will not be used anymore

        return video_metrics


def predictVideo(
    model,
    video_path : str | Path,
    save_path : str | Path | None = None,
    real_fake_pattern : list[str] = [],
    real : int = 2,
    fake : int = 1,
    frames_as_input : int = 2,
    w : int = 1280,
    h : int = 720):

    if not real_fake_pattern:
        real_fake_pattern = ["fake"] * fake + ["real"] * real
    
    pattern_len = len(real_fake_pattern)

    if save_path: save_path = str(save_path)
    video_path = str(video_path)
    
    # cap = cv2.VideoCapture(video_path)
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))

    if save_path:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(save_path, fourcc, fps, (h, w))

    empty_metric = get_metrics_example(0)

    video_metrics = {i : get_metrics_example() for i in range()}

    frames_iterator = getFrames(video_path, w, h)

    frames = [ # all frames
        next(frames_iterator) for i in range(frames_as_input)]

    for i in range(frames_as_input)
        for metric_name in empty_metric:
            video_metrics[metric_name].append(0)

    if save_path:
        for i in range(frames_as_input):
            writer.write(frames[i])

    for real in frames_iterator:            
        if real_fake_pattern[i % pattern_len] == "real": # read next frame
            frames.append(real)
            if save_path: writer.write(real)

            for metric_name in empty_metric:
                video_metrics[metric_name].append(0)
            
        else: # predict next frame
            fake = model.predict(frames[-frames_as_input:])
            frames.append(fake)
            if save_path: writer.write(fake)

            frame_metrics = compute_metrics(real, fake)

            for metric_name in frame_metrics:
                video_metrics[metric_name].append(frame_metrics[metric_name])

        del frames[0] # dont need it anymore

    if save_path: writer.release()

    return video_metrics



def predictVideoIterator(
    model,
    video_path : str | Path,
    save_path : str | Path | None = None,
    real_fake_pattern : list[str] = [],
    real : int = 2,
    fake : int = 1,
    frames_as_input : int = 2,
    w : int = 1280,
    h : int = 720):

    if not real_fake_pattern:
        real_fake_pattern = ["fake"] * fake + ["real"] * real
    
    pattern_len = len(real_fake_pattern)

    if save_path: save_path = str(save_path)
    video_path = str(video_path)
    
    # cap = cv2.VideoCapture(video_path)
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))

    if save_path:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(save_path, fourcc, fps, (h, w))

    empty_metric = get_metrics_example(-1)

    video_metrics = {i : get_metrics_example() for i in range()}

    frames_iterator = getFrames(video_path, w, h)

    frames = [ # all frames
        next(frames_iterator) for i in range(frames_as_input)]

    for i in range(frames_as_input)
        for metric_name in empty_metric:
            video_metrics[metric_name].append(-1)

    if save_path:
        for i in range(frames_as_input):
            writer.write(frames[i])

    for real in frames_iterator:            
        if real_fake_pattern[i % pattern_len] == "real": # read next frame
            frames.append(real)
            if save_path: writer.write(real)

            frame_metrics = empty_metric

            for metric_name in empty_metric:
                video_metrics[metric_name].append(-1)
            
        else: # predict next frame
            fake = model.predict(frames[-frames_as_input:])
            frames.append(fake)
            if save_path: writer.write(fake)

            frame_metrics = compute_metrics(real, fake)

            for metric_name in frame_metrics:
                video_metrics[metric_name].append(frame_metrics[metric_name])

        yield real, frames[-1], frame_metrics, video_metrics

        del frames[0] # dont need it anymore

    if save_path: writer.release()