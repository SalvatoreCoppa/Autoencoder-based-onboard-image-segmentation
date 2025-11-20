import torch

def compute_miou(pred, target, num_classes=9, ignore_index=0, eps=1e-7):
    pred = torch.argmax(pred, dim=1)
    miou = 0.0
    count = 0

    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()

        if union > 0:
            miou += (intersection + eps) / (union + eps)
            count += 1

    return miou / count if count > 0 else 0.0

def compute_pixel_accuracy(pred, target, ignore_index=0):
    pred = torch.argmax(pred, dim=1)
    mask = target != ignore_index
    correct = (pred == target) & mask
    return correct.sum().item() / mask.sum().item()

