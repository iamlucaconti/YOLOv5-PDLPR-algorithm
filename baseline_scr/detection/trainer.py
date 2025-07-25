import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import math
import random
import numpy as np
from torchvision.ops import complete_box_iou_loss

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1. CIoU‑loss per bbox (cx,cy,w,h) ∈ [0,1]
def cxcywh_to_xyxy(box):                 # box: B×4
    cx, cy, w, h = box.unbind(1)
    return torch.stack([cx - w / 2, cy - h / 2,
                        cx + w / 2, cy + h / 2], dim=1)

def ciou_loss(pred, tgt):
    """pred, tgt in the format cxcywh – return mean loss medium batch."""
    return complete_box_iou_loss(
        cxcywh_to_xyxy(pred), cxcywh_to_xyxy(tgt)
    ).mean()

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        preds = model(imgs)
        loss  = ciou_loss(preds, targets)
        loss.backward()
        optimizer.step()

        running += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        loss_sum += ciou_loss(preds, targets).item() * imgs.size(0)
    return loss_sum / len(loader.dataset)

def train(model, dl_train, dl_val,
          epochs: int, lr: float, device: str,
          ckpt_path: str | Path = "best_lpdet.pt"):

    model.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )

    # scheduler: each 5 epoch lr = lr* 0.8
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.8
    )

    best_val = math.inf
    for epoch in range(1, epochs + 1):
        print(f"\nStarting epoch {epoch}/{epochs}")
        tr_loss = train_one_epoch(model, dl_train, optimizer, device, epoch)
        val_loss = evaluate(model, dl_val, device) if dl_val else 0.0

        scheduler.step() 

        print(f" Epoch {epoch:02d}/{epochs} | train_loss: {tr_loss:.6f} "
              f"| val_loss: {val_loss:.6f} | lr: {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"     saved new best model  {ckpt_path}")

    return model