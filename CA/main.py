from .color import ImageColor
from .composition import Composition
from .texture import Texture
from .base import Base


class CA(Base):
    def __init__(self, image_path):
        self.color = ImageColor(image_path)
        self.composition = Composition(image_path)
        self.texture = Texture(image_path)

    def compute_ca(self):
        color_res = self.color.compute_color_info()
        composition_res = self.composition.compute_composition_info()
        texture_res = self.texture.compute_texture_info()

        return color_res + composition_res + texture_res

    def update(self, img_name):
        self.color.update(img_name)
        self.composition.update(img_name)
        self.texture.update(img_name)


if __name__ == "__main__":
    img_path = "../japan.png"
    ca = CA(img_path)
    res = ca.compute_ca()
    print(len(res))
    print(res)
