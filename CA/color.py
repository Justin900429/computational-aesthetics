import os
import numpy as np
import cv2
from .base import Base

SHOW = False
ROOT_DIR = os.path.dirname(__file__)

# Only create when going to plot image
if SHOW:
    POINTS = [32, 96, 160, 224]
    COORDINATE = []
    for point_x in POINTS:
        for point_y in POINTS:
            for point_z in POINTS:
                COORDINATE.append([point_x, point_y, point_z])
    COORDINATE = np.array(COORDINATE)


class ImageColor(Base):
    def __init__(self, image_path):
        super(ImageColor, self).__init__()

        # Read in the image
        self.bgr_img = None
        self.hsv_img = None
        self.gray_img = None

        self.h_mean, self.s_mean, self.v_mean = None, None, None
        self.h_std, self.s_std, self.v_std = None, None, None

        # Read in the image
        self.read_in(image_path)

    def update(self, image_path):
        self.read_in(image_path)

    def read_in(self, image_path):
        self.bgr_img = cv2.imread(image_path)
        self.hsv_img = cv2.cvtColor(self.bgr_img, cv2.COLOR_BGR2HSV)
        self.gray_img = cv2.imread(image_path, 0)

        # Compute the statics
        self.h_mean, self.s_mean, self.v_mean = np.mean(self.hsv_img, axis=(0, 1))
        self.h_std, self.s_std, self.v_std = np.std(self.hsv_img, axis=(0, 1))

    @staticmethod
    def compute_circular(channel_image):
        A = np.cos(channel_image).sum()
        B = np.sin(channel_image).sum()

        R = 1 - np.sqrt(A ** 2 + B ** 2) / (channel_image.shape[0] * channel_image.shape[1])

        return R

    def compute_hsv_statics(self):
        h_circular = self.compute_circular(self.hsv_img[0])
        v_intensity = np.sqrt((self.hsv_img[-1] ** 2).mean())

        return [self.s_mean, self.s_std, self.v_std, h_circular, v_intensity]

    def compute_emotion_based(self):
        valence = 0.69 * self.v_mean + 0.22 * self.s_mean
        arousal = -0.31 * self.v_mean + 0.6 * self.s_mean
        dominance = -0.76 * self.v_mean + 0.32 * self.s_mean

        return [valence, arousal, dominance]

    def compute_color_diversity(self):
        """Adapted from
        https://github.com/yilangpeng/computational-aesthetics/blob/27ff52b47b880bd46a14a7b062a4dde69b6a9988/basic.py#L46-L56
        """
        rgb = cv2.cvtColor(self.bgr_img, cv2.COLOR_BGR2RGB).astype(float)

        l_rgbR, l_rgbG, l_rgbB = cv2.split(rgb)
        l_rg = l_rgbR - l_rgbG
        l_yb = 0.5 * l_rgbR + 0.5 * l_rgbG - l_rgbB

        rg_sd = np.std(l_rg)
        rg_mean = np.mean(l_rg)
        yb_sd = np.std(l_yb)
        yb_mean = np.mean(l_yb)

        rg_yb_sd = (rg_sd ** 2 + yb_sd ** 2) ** 0.5
        rg_yb_mean = (rg_mean ** 2 + yb_mean ** 2) ** 0.5
        colorful = rg_yb_sd + (rg_yb_mean * 0.3)

        return [colorful]

    def compute_color_name(self):
        """Adapted from
        https://github.com/yilangpeng/computational-aesthetics/blob/master/colorname.py
        """
        # Load the color dict
        color_dict = np.load(os.path.join(ROOT_DIR, "color_dict.npz"))["color_dict"]

        rgb_image = cv2.cvtColor(self.bgr_img, cv2.COLOR_BGR2RGB)
        rgb_image = rgb_image.astype("int64")
        rgb_image = rgb_image // 8

        r_channel, g_channel, b_channel = cv2.split(rgb_image)
        one_key = r_channel * (32 * 32) + g_channel * 32 + b_channel
        tf = color_dict[one_key]

        # Compute the fraction for each color
        h, w = self.bgr_img.shape[:2]

        unique, counts = [list(x) for x in np.unique(tf, return_counts=True)]  # count colors
        color_percents = [0] * 11  # calculate the percentages of each basic color
        for i in range(0, 11):
            if i in unique:
                k = unique.index(i)
                color_percents[i] = counts[k] / (h * w)

        if SHOW:
            import matplotlib.pyplot as plt
            # RGB color for list named color
            color_list_rgb = [
                (0, 0, 0), (0, 0, 255), (128, 102, 64),  # black, blue, brown
                (128, 128, 128), (0, 255, 0), (255, 204, 0),  # gray, green, orange
                (255, 128, 255), (255, 0, 255), (255, 0, 0),  # pink, purple, red
                (255, 255, 255), (255, 255, 0)  # white, yellow
            ]

            # Create colored image using 11 colors
            color_image = np.zeros((h, w, 3), dtype='uint8')
            for idx in range(0, 11):
                mask = (tf == idx)
                color_image[mask] = color_list_rgb[idx]
            plt.imshow(color_image)
            plt.axis("off")
            plt.show()

        return color_percents

    def compute_color_info(self):
        hsv_res = self.compute_hsv_statics()
        emotion_res = self.compute_emotion_based()
        color_div_res = self.compute_color_diversity()
        color_name_res = self.compute_color_name()

        return hsv_res + emotion_res + color_div_res + color_name_res


if __name__ == "__main__":
    image = ImageColor("face.jpg")
    res = image.compute_color_diversity()
    print(res)
