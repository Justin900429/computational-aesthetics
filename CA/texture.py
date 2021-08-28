import math
import cv2
import numpy as np
import pywt
import skimage.measure
import skimage.feature
import skimage.morphology
import skimage.filters.rank
from .base import Base

SHOW = False


class Texture(Base):
    def __init__(self, image_path):
        super(Texture, self).__init__()
        self.bgr_img = None
        self.hsv_img = None
        self.gray_img = None
        self.read_in(image_path)

    def compute_entropy(self):
        entropy = skimage.filters.rank.entropy(image=self.gray_img,
                                               selem=skimage.morphology.square(9))
        entropy = entropy.mean()
        return [entropy]

    def update(self, image_path):
        self.read_in(image_path)

    def read_in(self, image_path):
        self.bgr_img = cv2.imread(image_path)
        self.hsv_img = cv2.cvtColor(self.bgr_img, cv2.COLOR_BGR2HSV)
        self.gray_img = cv2.imread(image_path, 0)

    @staticmethod
    def compute_channel_wavlet(wavelets):
        if SHOW:
            import matplotlib.pyplot as plt
            image, _ = pywt.coeffs_to_array(wavelets)
            plt.imshow(image, cmap="gray")
            plt.show()

        result_value = []
        level_mean_value = 0.0
        for wavelet in wavelets[1:]:
            wavelet_feature = 0.0
            magnitude = 0.0

            temp_level_mean_value = 0
            for level_list in wavelet:
                wavelet_feature += np.sum(level_list)
                magnitude += (level_list.shape[0] * level_list.shape[1])
                temp_level_mean_value += np.mean(level_list)

            level_mean_value += (temp_level_mean_value / 3)
            result_value.append(wavelet_feature / magnitude)

        result_value.append(level_mean_value)
        return result_value

    def compute_wavelet_texture(self):
        # Process for the H channel
        h_wavelet = pywt.wavedec2(self.hsv_img[..., 0], mode="periodization",
                                  wavelet="db2", level=3)
        h_features = self.compute_channel_wavlet(h_wavelet)

        # Process for the S channel
        s_wavelet = pywt.wavedec2(self.hsv_img[..., 1], mode="periodization",
                                  wavelet="db2", level=3)
        s_features = self.compute_channel_wavlet(s_wavelet)

        # Process for the V channel
        v_wavelet = pywt.wavedec2(self.hsv_img[..., 2], mode="periodization",
                                  wavelet="db2", level=3)
        v_features = self.compute_channel_wavlet(v_wavelet)

        return h_features + s_features + v_features

    def coarseness(self, max_k=5):
        """Adapted from
        https://github.com/Sdhir/TamuraFeatures/blob/f22f99a5898b2414e1c3f89d464a3a761e8b6a98/Tamura.m#L26-L191
        https://github.com/MarshalLeeeeee/Tamura-In-Python/blob/1a079acce55989d65fb1a676e0bf6b9fbe54be82/tamura-numpy.py#L4-L37
        """
        gray_img = np.copy(self.gray_img)
        h, w = gray_img.shape
        if 2 ** max_k >= w or 2 ** max_k >= h:
            max_k = min(int(math.log(h) / math.log(2)), int(math.log(w) / math.log(2)))

        average_gray = np.zeros((max_k, h, w))
        for k in range(1, max_k + 1):
            for row in range(2 ** (k-1), h - 2**(k-1)):
                for col in range(2 ** (k-1), w - 2**(k-1)):
                    if len(gray_img[row - 2**(k-1):row + 2**(k-1), col - 2**(k-1):col + 2**(k-1)]) == 0:
                        assert 1 == 2
                    average_gray[k - 1, row, col] = \
                        gray_img[row - 2**(k-1):row + 2**(k-1) + 1, col - 2**(k-1):col + 2**(k-1) + 1].mean()

        expected_horizontal = np.zeros((max_k, h, w))
        expected_vertical = np.zeros((max_k, h, w))

        for k in range(1, max_k + 1):
            for row in range(2**(k-1), h - 2**(k-1)):
                for col in range(2**(k-1), w - 2**(k-1)):
                    expected_horizontal[k - 1, row, col] = \
                        np.abs(
                            average_gray[k - 1, row + 2**(k-1), col] - average_gray[k - 1, row - 2**(k-1), col])
                    expected_vertical[k - 1, row, col] = \
                        np.abs(
                            average_gray[k - 1, row, col + 2**(k-1)] - average_gray[k - 1, row, col - 2**(k-1)])

        coarseness_best = np.zeros((h, w))
        for row in range(h):
            for col in range(w):
                max_horizontal = np.max(expected_horizontal[:, row, col])
                argmax_horizontal = np.argmax(expected_horizontal[:, row, col])
                max_vertical = np.max(expected_vertical[:, row, col])
                argmax_vertical = np.argmax(expected_vertical[:, row, col])

                if max_horizontal > max_vertical:
                    max_arg_k = argmax_horizontal
                else:
                    max_arg_k = argmax_vertical

                coarseness_best[row, col] = 2 ** max_arg_k

        coarseness = coarseness_best.mean()

        return coarseness

    def contrast(self):
        """Adapted from
        https://github.com/Sdhir/TamuraFeatures/blob/f22f99a5898b2414e1c3f89d464a3a761e8b6a98/Tamura.m#L194-L206
        """
        gray_img = np.copy(self.gray_img).reshape(-1)
        average_value = np.mean(gray_img)

        base_value = gray_img - average_value
        fourth_moment = np.mean(np.power(base_value, 4))
        variance = np.mean(np.power(base_value, 2))

        alpha = fourth_moment / (variance ** 2)
        contrast_value = math.sqrt(variance) / math.pow(alpha, 0.25)

        return contrast_value

    def directionality(self):
        """Adapted from
        https://github.com/Sdhir/TamuraFeatures/blob/f22f99a5898b2414e1c3f89d464a3a761e8b6a98/Tamura.m#L209-L302
        """
        # Padding image for the filter
        gray_img = np.copy(self.gray_img).astype("int64")
        h, w = gray_img.shape[:2]

        horizontal_filter = np.array([[-1, 0, 1],
                                      [-1, 0, 1],
                                      [-1, 0, 1]])
        vertical_filter = np.array([[1, 1, 1],
                                    [0, 0, 0],
                                    [-1, -1, -1]])

        # Applying horizontal pattern filter
        delta_horizontal = cv2.filter2D(src=gray_img.astype(np.float), ddepth=-1,
                                        kernel=horizontal_filter)
        for wi in range(0, w - 1):
            delta_horizontal[0][wi] = gray_img[0][wi + 1] - gray_img[0][wi]
            delta_horizontal[h - 1][wi] = gray_img[h - 1][wi + 1] - gray_img[h - 1][wi]
        for hi in range(0, h):
            delta_horizontal[hi][0] = gray_img[hi][1] - gray_img[hi][0]
            delta_horizontal[hi][w - 1] = gray_img[hi][w - 1] - gray_img[hi][w - 2]

        # Applying vertical pattern filter
        delta_vertical = cv2.filter2D(src=gray_img.astype(np.float), ddepth=-1,
                                      kernel=vertical_filter)
        for wi in range(0, w):
            delta_vertical[0][wi] = gray_img[1][wi] - gray_img[0][wi]
            delta_vertical[h - 1][wi] = gray_img[h - 1][wi] - gray_img[h - 2][wi]
        for hi in range(0, h - 1):
            delta_vertical[hi][0] = gray_img[hi + 1][0] - gray_img[hi][0]
            delta_vertical[hi][w - 1] = gray_img[hi + 1][w - 1] - gray_img[hi][w - 1]

        delta_magnitude = (np.abs(delta_horizontal) + np.abs(delta_vertical)) / 2.0
        delta_magnitude_vec = delta_magnitude.reshape(-1)

        # Calculate the angle (theta)
        theta = np.zeros([h, w])
        for row in range(h):
            for col in range(w):
                if delta_horizontal[row][col] == 0 and delta_vertical[row][col] == 0:
                    theta[row][col] = 0
                elif delta_horizontal[row][col] == 0:
                    theta[row][col] = np.pi
                else:
                    theta[row][col] = np.arctan(delta_vertical[row][col] / delta_horizontal[row][col]) + np.pi / 2.0
        theta_vec = theta.reshape(-1)

        n = 16
        thresh = 12
        HD = np.zeros(n)

        magnitude_len = delta_magnitude_vec.shape[0]
        for ni in range(n):
            for k in range(magnitude_len):
                if (delta_magnitude_vec[k] >= thresh) and \
                   (theta_vec[k] >= (2 * ni - 1) * np.pi / (2 * n)) and \
                   (theta_vec[k] < (2 * ni + 1) * np.pi / (2 * n)):
                    HD[ni - 1] += 1
        HD = HD / np.sum(HD)

        directionality = 0.0
        HD_max_index = np.argmax(HD)
        for ni in range(n):
            directionality += np.power((ni - HD_max_index), 2) * HD[ni]

        return directionality

    def compute_tamura(self):
        coarseness = self.coarseness()
        contrast = self.contrast()
        directionality = self.directionality()

        return [coarseness, contrast, directionality]

    @staticmethod
    def get_features_by_channels(channel):
        contrast = skimage.feature.greycoprops(channel, "contrast")
        correlation = skimage.feature.greycoprops(channel, "correlation")
        energy = skimage.feature.greycoprops(channel, "energy")
        homogeneity = skimage.feature.greycoprops(channel, "homogeneity")

        return [contrast.sum(), correlation.sum(), energy.sum(), homogeneity.sum()]

    def compute_glcm_features(self):
        # Compute the value for gray channel
        gray_channel = skimage.feature.greycomatrix(self.gray_img, [1], [0], normed=True)
        features = self.get_features_by_channels(gray_channel)

        return features

    def compute_texture_info(self):
        entropy_res = self.compute_entropy()
        wavlet_res = self.compute_wavelet_texture()
        tamura_res = self.compute_tamura()
        glcm_res = self.compute_glcm_features()

        return entropy_res + wavlet_res + tamura_res + glcm_res


if __name__ == "__main__":
    texture = Texture("H_hom_1.png")
    results = texture.compute_glcm_features()
    print(results)
