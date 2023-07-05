from model.models import Model, DMVFN

if __name__ == '__main__':
    model = Model(DMVFN(load_path = "model/pretrained_models/dmvfn_city.pkl", device="cpu"))
    model.predictVideo(video_path='C:/Users/tuman/Desktop/test_clip/test_2.mp4', real=1, fake=1, save_path=None)
