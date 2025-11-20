import os
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

from utils.split_utils import compute_distribution, compute_presence_distribution

NUM_CLASSES = 9  # classi da 0 a 8 (dove 0 = Other)

def analyze_class_distribution(dataset_dir):
    """
    Analizza la distribuzione delle classi nel dataset.
    Restituisce un DataFrame con percentuale di pixel e percentuale di presenza per classe.
    """
    total_images = 0
    pixel_counts = defaultdict(int)
    class_presence_counts = defaultdict(int)

    for folder in tqdm(sorted(os.listdir(dataset_dir)), desc="Analyzing"):
        label_path = os.path.join(dataset_dir, folder, 'labels.png')
        if not os.path.isfile(label_path):
            continue

        total_images += 1
        label = np.array(Image.open(label_path), dtype=np.uint8).flatten()
        unique_classes = np.unique(label)

        for cls in unique_classes:
            count = np.sum(label == cls)
            pixel_counts[cls] += count
            class_presence_counts[cls] += 1

    total_pixels = sum(pixel_counts.values())

    rows = []
    for cls in range(NUM_CLASSES):
        abs_pixels = pixel_counts[cls]
        perc_pixels = abs_pixels / total_pixels if total_pixels else 0
        presence_freq = class_presence_counts[cls] / total_images if total_images else 0
        rows.append({
            'Class': cls,
            'Pixel %': round(perc_pixels * 100, 2),
            'Presence % (images)': round(presence_freq * 100, 2)
        })

    df = pd.DataFrame(rows)
    return df


def analyze_images_distribution(dataset_dir, subset=None):
    """
    Analizza le immagini di un dataset (o solo un sottoinsieme) e restituisce:
    - Distribuzioni pixel per immagine
    - Classi presenti per immagine
    - Distribuzione globale dei pixel
    - Frequenza di presenza globale per ogni classe
    - Lista delle immagini valide analizzate
    """
    image_pixel_distributions = {}
    image_class_presence = {}
    global_pixel_counts = defaultdict(int)
    global_class_presence = defaultdict(int)

    print("Analisi immagini...")
    all_images = subset if subset is not None else sorted(os.listdir(dataset_dir))
    
    for folder in tqdm(all_images):
        label_path = os.path.join(dataset_dir, folder, "labels.png")
        if not os.path.isfile(label_path):
            continue

        label = np.array(Image.open(label_path), dtype=np.uint8).flatten()
        unique_classes = np.unique(label)

        pixel_dist = defaultdict(int)
        class_present = set()

        for cls in unique_classes:
            count = np.sum(label == cls)
            if count > 0:
                pixel_dist[cls] += count
                global_pixel_counts[cls] += count
                global_class_presence[cls] += 1
                class_present.add(cls)

        image_pixel_distributions[folder] = pixel_dist
        image_class_presence[folder] = class_present

    total_images = len(image_pixel_distributions)
    images_list = list(image_pixel_distributions.keys())
    global_pixel_dist = compute_distribution(global_pixel_counts, list(range(NUM_CLASSES)))
    global_presence_dist = compute_presence_distribution(global_class_presence, total_images, list(range(NUM_CLASSES)))

    return (
        image_pixel_distributions,
        image_class_presence,
        global_pixel_dist,
        global_presence_dist,
        images_list
    )
