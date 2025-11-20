from collections import defaultdict

def compute_distribution(counts, class_names):
    total = sum(counts.values())
    return {cls: counts[cls] / total if total else 0.0 for cls in class_names}

def compute_presence_distribution(presence_counts, total_imgs, class_names):
    return {cls: presence_counts[cls] / total_imgs if total_imgs else 0.0 for cls in class_names}

def compute_distance(d1, d2, class_names):
    return sum(abs(d1[c] - d2[c]) for c in class_names)

def greedy_split(image_pixel_distributions, image_class_presence, global_pixel_distribution, global_presence_distribution, class_names, test_ratio):
    total_images = len(image_pixel_distributions)
    target_test_size = int(total_images * test_ratio)
    selected_test_set = []
    remaining_images = set(image_pixel_distributions.keys())
    test_pixel_counts = defaultdict(int)
    test_class_presence = defaultdict(int)

    while len(selected_test_set) < target_test_size:
        best_candidate = None
        best_distance = float('inf')

        for img in remaining_images:
            temp_pixel_counts = test_pixel_counts.copy()
            for cls in image_pixel_distributions[img]:
                temp_pixel_counts[cls] += image_pixel_distributions[img][cls]
            pixel_dist = compute_distribution(temp_pixel_counts, class_names)
            pixel_score = compute_distance(pixel_dist, global_pixel_distribution, class_names)

            temp_presence_counts = test_class_presence.copy()
            for cls in image_class_presence[img]:
                temp_presence_counts[cls] += 1
            presence_dist = compute_presence_distribution(temp_presence_counts, len(selected_test_set)+1, class_names)
            presence_score = compute_distance(presence_dist, global_presence_distribution, class_names)

            total_score = 0.5 * pixel_score + 0.5 * presence_score

            if total_score < best_distance:
                best_distance = total_score
                best_candidate = img

        if best_candidate:
            selected_test_set.append(best_candidate)
            remaining_images.remove(best_candidate)
            for cls in image_pixel_distributions[best_candidate]:
                test_pixel_counts[cls] += image_pixel_distributions[best_candidate][cls]
            for cls in image_class_presence[best_candidate]:
                test_class_presence[cls] += 1
        else:
            break

    return selected_test_set, list(remaining_images), test_pixel_counts, test_class_presence
