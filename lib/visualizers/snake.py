from lib.utils.snake import snake_config
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import colorsys
import random

mean = snake_config.mean
std = snake_config.std

class Visualizer:
    def get_colors(self):
        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [126, 126, 126],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)
        return colors

    def random_colors(self, N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def visualize_gts(self, image, polys, save_path=None):
        auto_show = False

        colors = self.random_colors(len(polys))
        fig, ax = plt.subplots(1, figsize=(10, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(image)

        for i in range(len(polys)):
            color = colors[i]
            poly = polys[i]
            poly = np.append(poly, [poly[0]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], color=color, linestyle='-', linewidth=4)
            ax.scatter(poly[:, 0], poly[:, 1], marker='o', color=color, s=60)

        if save_path is not None:
            plt.savefig(save_path)
        if auto_show:
            plt.show()
        plt.close()

    def visualize_preds(self, image, polys, save_path=None):
        auto_show = False
        colors = self.random_colors(len(polys))

        fig, ax = plt.subplots(1, figsize=(10, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(image)

        for i in range(len(polys)):
            color = colors[i]
            poly = polys[i]
            if len(poly) > 0:
                poly = np.append(poly, [poly[0]], axis=0)
                ax.plot(poly[:, 0], poly[:, 1], color=color, linestyle='-', marker='o', linewidth=4)

        if save_path is not None:
            plt.savefig(save_path)
        if auto_show:
            plt.show()
        plt.close()