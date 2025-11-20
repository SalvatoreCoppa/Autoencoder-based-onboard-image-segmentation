import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Percentuali pixel delle classi 1–8 (escludendo la classe 0)
PIXEL_PERCENT = np.array([15.04, 13.27, 15.11, 0.28, 1.11, 6.23, 35.91, 10.16])

def get_class_weights(pixel_percents=PIXEL_PERCENT, normalize=True, eps=1e-6):
    pixel_percents = pixel_percents / 100.0  # Converti da percentuale a proporzione
    weights = 1.0 / np.log(pixel_percents + eps)
    if normalize:
        weights = weights / weights.sum()
    weights = np.insert(weights, 0, 0.0)  # Aggiungi peso zero per la classe 0 (ignorata)
    return torch.tensor(weights, dtype=torch.float32)

def get_weighted_crossentropy_loss(device, pixel_percents=PIXEL_PERCENT):
    weights = get_class_weights(pixel_percents=pixel_percents).to(device)
    return nn.CrossEntropyLoss(weight=weights, ignore_index=0)

def jaccard_loss(pred, target, num_classes=9, ignore_index=0, eps=1e-7):
    pred = F.softmax(pred, dim=1)  # [B, C, H, W]
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    loss = 0.0
    count = 0

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        pred_cls = pred[:, cls, :, :]
        target_cls = target_one_hot[:, cls, :, :]

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        iou = (intersection + eps) / (union + eps)

        loss += 1 - iou
        count += 1

    return loss / count if count > 0 else torch.tensor(0.0, device=pred.device)

def combined_loss(pred, target, ce_loss_fn, alpha=1, num_classes=9):
    ce = ce_loss_fn(pred, target)
    jac = jaccard_loss(pred, target, num_classes=num_classes, ignore_index=0)
    return alpha * ce + (1 - alpha) * jac