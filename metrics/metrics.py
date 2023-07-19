import numpy as np
import cv2
import math

# metrics
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity
from scipy.spatial.distance import hamming

# torch metrics
import lpips
from pytorch_msssim import MS_SSIM

# for erqa
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import f1_score


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

lpips_model = lpips.LPIPS(net='alex').to(DEVICE)
lpips_model.eval()

ms_ssim_module = MS_SSIM(data_range=255, size_average=False, channel=3)
ms_ssim_module = ms_ssim_module.to(DEVICE)
ms_ssim_module.eval()

ms_ssim_module_1channel = MS_SSIM(data_range=255, size_average=False, channel=1)
ms_ssim_module_1channel = ms_ssim_module_1channel.to(DEVICE)
ms_ssim_module_1channel.eval()


def lpips_metric(image1, image2):
    with torch.no_grad():
        image1 = lpips.im2tensor(image1).to(DEVICE)
        image2 = lpips.im2tensor(image2).to(DEVICE)

        score = lpips_model(image1, image2).squeeze().cpu().numpy()

    return 1 - score


def ms_ssim_metric(image1, image2):
    with torch.no_grad():
        image1 = image1.transpose(2, 0, 1)
        image2 = image2.transpose(2, 0, 1)

        image1 = torch.tensor(image1, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        image2 = torch.tensor(image2, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        score = ms_ssim_module(image1, image2).cpu().squeeze().numpy()

    return float(score)


def yuv_ms_ssim_metric(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2YUV)[:, :, 0]
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2YUV)[:, :, 0]

    with torch.no_grad():
        image1 = torch.tensor(image1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        image2 = torch.tensor(image2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        score = ms_ssim_module_1channel(image1, image2).cpu().squeeze().numpy()

    return float(score)



def cosine_similarity_metric(image1, image2):

    image1_vector = np.array(image1).ravel()
    image2_vector = np.array(image2).ravel()

    similarity_score = cosine_similarity(
        [image1_vector], [image2_vector])[0][0]

    return similarity_score


def hamming_distance_metric(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    hamming_distance = hamming(image1.flatten(), image2.flatten())

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


def yuv_psnr_metric(image1, image2):
    image1_y_component = cv2.cvtColor(image1, cv2.COLOR_RGB2YUV)[:, :, 0]
    image2_y_component = cv2.cvtColor(image2, cv2.COLOR_RGB2YUV)[:, :, 0]

    score = psnr_metric(image1_y_component, image2_y_component)

    return score


def ssim_metric(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    score = structural_similarity(
        image1, image2, data_range=(image2.max() - image2.min()))

    return score


def yuv_ssim_metric(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2YUV)[:, :, 0]
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2YUV)[:, :, 0]

    score = structural_similarity(
        image1, image2, data_range=(image2.max() - image2.min()))

    return score


def erqa_metric(image1, image2):
    ### TODO there is also an official implementation of this metric, should add it as well
    real = cv2.Canny(image1, 200, 100) == 255
    fake = cv2.Canny(image2, 200, 100) == 255

    # To compensate for the one-pixel shift, edges that are no more than one pixel away from the GT's are considered true-positive
    adjusted = sliding_window_view(real, (3, 3)).sum((-1, -2))
    adjusted = adjusted > 0

    adjusted = adjusted.flatten()
    
    # adjusted lose 1 pixel at the begging and end of each axes, due to sliding window
    fake_adjusted = fake[1:-1, 1:-1].flatten()
    fake = fake.flatten()
    real = real.flatten()

    # calculate f1
    tp = np.sum(fake_adjusted & adjusted)
    fp = np.sum(fake & (~real))
    fn = np.sum(real & (~fake))

    return tp / (tp + 0.5 * (fp + fn))


### TODO some metrics work faster with batches, should probably pass batches here
def compute_metrics(img1, img2):
    return {
        "cossim" : cosine_similarity_metric(img1, img2),
        "hamming" : hamming_distance_metric(img1, img2),
        "mse" : mse_metric(img1, img2),
        "psnr" : psnr_metric(img1, img2),
        "ssim" : ssim_metric(img1, img2),
        "yuv_psnr" : yuv_psnr_metric(img1, img2),
        "yuv_ssim" : yuv_ssim_metric(img1, img2),
        "lpips" : lpips_metric(img1, img2),
        "ms_ssim" : ms_ssim_metric(img1, img2),
        "yuv_ms_ssim" : yuv_ms_ssim_metric(img1, img2),
        "erqa" : erqa_metric(img1, img2),
    }


def get_metrics_example():
    return {
        "cossim" : [],
        "hamming" : [],
        "mse" : [],
        "psnr" : [],
        "ssim" : [],
        "yuv_psnr" : [],
        "yuv_ssim" : [],
        "lpips" : [],
        "ms_ssim" : [],
        "yuv_ms_ssim" : [],
        "erqa" : [],
    }