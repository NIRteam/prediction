import torch
import cv2
# from tqdm.notebook import tqdm
from tqdm import tqdm
import numpy as np

import traceback
import os
from copy import deepcopy

from DMVFN.model.arch import *


class Model():
    def __init__(self, load_path, device="cuda"):
        self.device = device
        self.dmvfn = DMVFN()
        self.dmvfn.to(device)

        state_dict = torch.load(load_path)
        model_state_dict = self.dmvfn.state_dict()
        for k in model_state_dict.keys():
            model_state_dict[k] = state_dict['module.'+k]
        self.dmvfn.load_state_dict(model_state_dict)

    def eval(self):
        self.dmvfn.eval()

    def evaluate(self, imgs : list[np.ndarray], scale_list : list[int] = [4,4,4,2,2,2,1,1,1]):
        with torch.no_grad():
            for i in range(len(imgs)):
                imgs[i] = torch.tensor(imgs[i].transpose(2, 0, 1).astype('float32'))
            
            img = torch.cat(imgs, dim=0)
            img = img.unsqueeze(0) # .unsqueeze(0) # NCHW
            img = img.to(self.device, non_blocking=True) / 255.

            try:
                pred = self.dmvfn(img, scale=scale_list, training=False) # 1CHW
            except Exception:
                print(traceback.format_exc())
                raise RuntimeError("Image resolution is not compatible")

            if len(pred) == 0:
                pred = imgs[:, 0]
            else:
                pred = pred[-1]
            
            pred = np.array(pred.cpu().squeeze() * 255).transpose(1, 2, 0) # CHW -> HWC
            pred = pred.astype("uint8")
            
            return pred

    def predict(self, imgs : list[np.ndarray], num_frames_to_predict : int = 1) -> list[np.ndarray]:
        """
        imgs : list[np.ndarray] - фреймы, по которым будет происходить предсказание
        num_frames_to_predict : int - количество кадров, которое нужно предсказать
        """

        imgs = list(deepcopy(imgs)) ### do not modify the input list

        if num_frames_to_predict == 1:
            return self.evaluate(imgs)

        # how many frames are passed to model
        frames_to_model = len(imgs)

        for i in range(num_frames_to_predict):
            img_pred = self.evaluate(imgs[-frames_to_model:])
            imgs.append(img_pred)

        return imgs


    def predictVideo(self, video_path : str, save_path : str,
                     real : int = 1, fake : int = 1, frames_to_model : int = 2,
                     w : int = 1024, h : int = 1792,):
        """
        video_path : str - Путь до видео, фреймы в котором надо предсказать
        save_path : str or None - Куда сохранить новое видео, если None, то видео сохраняться не будет
        real : int - Количество кадров, которые считываются с реалього видео
        fake : int - Количество кадров, которые будут предсказаны
        frames_to_model : int - Сколько кадров передаётся в модель для предсказания
        w : int - Ширина видео
        h : int - Высота видео

        returns:
        frames : list[np.array] - все кадры видео
        real_fake_mask : list[str] - маска для frames, где реальные кадры отмечены "real", предсказанные кадры "fake"

        Примеры использования:
        >>> frames, mask = predictVideo("path/to/video", "where/to/save", real = 2, fake = 1, frames_to_model = 2)
        >>> mask
        ["real", "real", "fake", "real", "real", "fake", "real", "real", "fake", ...]
        >> len(mask) / mask.count("real") # 

        >>> frames, mask = predictVideo("path/to/video", "where/to/save", real = 1, fake = 1, frames_to_model = 2)
        >>> mask
        ["real", "real", "fake", "real",  "fake", "real", "fake", ...]

        >>> frames, mask = predictVideo("path/to/video", "where/to/save", real = 2, fake = 2, frames_to_model = 4)
        >>> mask
        ["real", "real", "real", "real", "fake", "fake", "real", "real", "fake", "fake", "real", "real", ...]
        """
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(save_path, fourcc, fps, (h, w))

        frames : list[np.ndarray] = []
        real_fake_mask : list[str] = []

        for i in range(frames_to_model):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (h, w))
                frames.append(frame)
                real_fake_mask.append("real")
                writer.write(frame)
            else:
                cap.release()
                writer.release()
                raise RuntimeError("Ошибка считывания видео")

        predicted_frames_num = 0

        try:
            for i in tqdm(range(frame_count-frames_to_model)):
                ret, frame = cap.read()

                if ret:
                    frame = cv2.resize(frame, (h, w))
                    
                    if predicted_frames_num >= fake: # read real frame
                        predicted_frames_num += 1

                        frames.append(frame)
                        real_fake_mask.append("real")
                        writer.write(frame)
                        if predicted_frames_num >= (fake + real):
                            predicted_frames_num = 0
                    
                    else: # predict next frame
                        predicted_frames_num += 1
                        
                        img_pred = self.predict(frames[-frames_to_model:])
                        writer.write(img_pred)
                        frames.append(img_pred)
                        real_fake_mask.append("fake")

                else:
                    break
        
        except Exception:
            print(traceback.format_exc())

        finally: # Закрываем файлы
            cap.release()
            writer.release()

        return frames, real_fake_mask


