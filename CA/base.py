from abc import ABC, abstractmethod


class Base(ABC):
    @abstractmethod
    def update(self, image_path):
        pass
