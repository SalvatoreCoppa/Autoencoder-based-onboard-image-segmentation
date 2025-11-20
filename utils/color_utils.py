from PIL import Image
import numpy as np

ID_TO_COLOR = {
    0: (255, 255, 255),  # Other
    1: (178, 176, 153),  # Smooth trail
    2: (128, 255, 0),  # Traversable grass
    3: (156, 76, 30),  # Rough trail
    4: (255, 0, 128),  # Puddle
    5: (255, 0, 0),  # Obstacle
    6: (0, 160, 0),  # Low vegetation
    7: (40, 80, 0),  # High vegetation
    8: (1, 88, 255),  # Sky
}


def save_indexed_label_with_palette(label_2d: np.ndarray, save_path: str):
    """Salva una label come PNG a indice con palette visiva corretta"""
    assert label_2d.ndim == 2
    palette_img = Image.fromarray(label_2d.astype(np.uint8), mode="P")

    # Costruisci la palette in formato flat (768 valori = 256*3)
    palette = [0] * 768
    for idx, color in ID_TO_COLOR.items():
        palette[idx * 3 + 0] = color[0]
        palette[idx * 3 + 1] = color[1]
        palette[idx * 3 + 2] = color[2]

    palette_img.putpalette(palette)
    palette_img.save(save_path)