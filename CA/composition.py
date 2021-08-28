import functools
import operator
import numpy as np
import pywt
from sklearn.cluster import (
    MeanShift,
    estimate_bandwidth
)
import cv2
from .base import Base

SHOW = False


class Composition(Base):
    def __init__(self, image_path):
        super(Composition, self).__init__()
        self.bgr_img = None
        self.hsv_img = None
        self.gray_img = None
        self.read_in(image_path)

    def compute_edge_pixels(self,
                            blur_size: int = 3,
                            ratio_low=0.4, ratio_up=0.8):
        """Adapted from
        https://github.com/yilangpeng/computational-aesthetics/blob/master/edge.py
        """
        h, w = self.bgr_img.shape[:2]
        blur_img = cv2.GaussianBlur(self.gray_img, (blur_size, blur_size), 0)

        thresh_low = min(100, np.quantile(blur_img, q=ratio_low))
        thresh_up = max(200, np.quantile(blur_img, q=ratio_up))

        edges_img = cv2.Canny(blur_img,
                              threshold1=thresh_low,
                              threshold2=thresh_up)
        num_edges = np.count_nonzero(edges_img) / (h * w)

        if SHOW:
            import matplotlib.pyplot as plt
            plt.imshow(edges_img, cmap="gray")
            plt.axis("off")
            plt.show()

        return [num_edges]

    def update(self, image_path):
        self.read_in(image_path)

    def read_in(self, image_path):
        self.bgr_img = cv2.imread(image_path)
        self.hsv_img = cv2.cvtColor(self.bgr_img, cv2.COLOR_BGR2HSV)
        self.gray_img = cv2.imread(image_path, 0)

    def compute_level_of_details(self,
                                 quantile=0.2,
                                 n_samples=3000,
                                 thresh=0.05):
        # Using quick shift segmentation
        rgb_img = cv2.cvtColor(self.bgr_img, cv2.COLOR_BGR2RGB)

        # Flatten the image
        flat_image = rgb_img.reshape(-1, 3)
        flat_image = np.float32(flat_image)

        bandwidth = estimate_bandwidth(flat_image,
                                       quantile=quantile,
                                       n_samples=n_samples)
        mean_shift = MeanShift(bandwidth, bin_seeding=True)
        mean_shift.fit(flat_image)
        image_labels = mean_shift.labels_

        h, w = rgb_img.shape[:2]
        unique_labels, unique_counts = np.unique(image_labels,
                                                 return_counts=True)

        if SHOW:
            import matplotlib.pyplot as plt
            # Get the average color of each segment
            total = np.zeros((len(unique_labels), 3), dtype=float)
            count = np.zeros_like(total, dtype=float)
            for i, label in enumerate(image_labels):
                total[label] = total[label] + flat_image[i]
                count[label] += 1
            avg = total / count
            avg = np.uint8(avg)

            # cast the labeled image into the corresponding average color
            res = avg[image_labels]
            result = res.reshape(rgb_img.shape)
            plt.imshow(result)
            plt.axis("off")
            plt.show()

        # Remove small region of image
        mean_thresh = (h * w) * thresh
        unique_labels = unique_labels[unique_counts > mean_thresh]
        unique_counts = unique_counts[unique_counts > mean_thresh]

        num_seg = len(unique_labels)
        average_size = unique_counts.mean() / (h * w)

        return [num_seg, average_size]

    @staticmethod
    def compute_channel_depth_of_field(channel):
        level_wanted = channel[1]
        h, w = level_wanted[0].shape[:2]

        # Create blank image to include all channel
        blank_image = np.zeros((h, w, 3))

        for idx, level_wanted_matrix in enumerate(level_wanted):
            blank_image[..., idx] = np.abs(level_wanted_matrix)

        if SHOW:
            import matplotlib.pyplot as plt
            plt.subplot(2, 2, 1)
            plt.imshow(np.abs(level_wanted[0]), cmap="gray")
            plt.subplot(2, 2, 2)
            plt.imshow(np.abs(level_wanted[1]), cmap="gray")
            plt.subplot(2, 2, 3)
            plt.imshow(np.abs(level_wanted[2]), cmap="gray")
            plt.subplot(2, 2, 4)
            plt.imshow(blank_image.mean(axis=-1), cmap="gray")
            plt.show()

        # Compute M6, M7, M10, M11
        start_x, end_x = int(h / 4), int(3 * h / 4)
        start_y, end_y = int(w / 4), int(3 * w / 4)

        dof_channel = np.sum(blank_image[start_x:end_x, start_y:end_y, :]) / np.sum(blank_image)

        return dof_channel

    def compute_depth_of_field(self):
        # Process for the H channel
        h_wavelet = pywt.wavedec2(self.hsv_img[..., 0], mode="periodization",
                                  wavelet="db3", level=3)
        h_dof = self.compute_channel_depth_of_field(h_wavelet)

        # Process for the S channel
        s_wavelet = pywt.wavedec2(self.hsv_img[..., 1], mode="periodization",
                                  wavelet="db3", level=3)
        s_dof = self.compute_channel_depth_of_field(s_wavelet)

        # Process for the V channel
        v_wavelet = pywt.wavedec2(self.hsv_img[..., 2], mode="periodization",
                                  wavelet="db3", level=3)
        v_dof = self.compute_channel_depth_of_field(v_wavelet)

        return [h_dof, s_dof, v_dof]

    def compute_rule_of_third(self):
        h, w = self.bgr_img.shape[:2]

        # Convert to hsv
        hsv_img = cv2.cvtColor(self.bgr_img, cv2.COLOR_BGR2HSV)

        # Set the end points of image
        start_h, end_h = int(h / 3), int(2 * h / 3)
        start_w, end_w = int(w / 3), int(2 * w / 3)
        center_image = hsv_img[start_h:end_h, start_w:end_w]

        # Compute the mean of saturation and value
        s_mean = np.mean(center_image[..., 1])
        v_mean = np.mean(center_image[..., 2])

        return [s_mean, v_mean]

    def compute_image_size(self):
        return [functools.reduce(operator.mul, self.bgr_img.shape)]

    def compute_composition_info(self):
        edge_res = self.compute_edge_pixels()
        LOD_res = self.compute_level_of_details()
        DOF_res = self.compute_depth_of_field()
        rule_of_thirds_res = self.compute_rule_of_third()
        img_size_res = self.compute_image_size()

        return edge_res + LOD_res + DOF_res + rule_of_thirds_res + img_size_res


if __name__ == "__main__":
    composition = Composition("face.jpg")
    count_edges = composition.compute_depth_of_field()
    print(count_edges)
