import numpy as np
from src.utils import rotate_image


def test_rotate_image_90():
    img = np.zeros((10, 20, 3), dtype=np.uint8)  # h=10, w=20
    rotated = rotate_image(img, 90)
    assert rotated.shape[0] == 20 and rotated.shape[1] == 10
