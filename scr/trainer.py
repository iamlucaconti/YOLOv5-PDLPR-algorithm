import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import time

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def character_accuracy(preds, targets):
    """Calcola la accuratezza a livello di carattere."""
    correct = 0
    total = 0
    for p, t in zip(preds, targets):
        correct += sum(pc == tc for pc, tc in zip(p, t))
        total += len(t)
    return correct / total if total > 0 else 0.0


def sequence_accuracy(preds, targets):
    """Calcola la accuratezza a livello di sequenza intera."""
    correct = sum(p == t for p, t in zip(preds, targets))
    return correct / len(targets)

def train(train_loader,
          val_loader,
          model,
          char2idx,
          device='cuda',
          num_epochs=10,
          lr=1e-3,
          load_checkpoint_path=None,
          save_checkpoint_path=None,
          lr_decay_factor=0.9,
          lr_decay_epochs=20):

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    blank_idx = char2idx['-']
    ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    start_epoch = 0
    best_val_loss = float('inf')
    last_decay_epoch = 0

    # Salvataggio delle perdite per ogni epoca
    train_losses = []
    val_losses = []

    # Caricamento checkpoint se disponibile
    if load_checkpoint_path and os.path.isfile(load_checkpoint_path):
        checkpoint = torch.load(load_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['weights'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_loss', best_val_loss)
        last_decay_epoch = checkpoint.get('last_decay_epoch', 0)
        print(f"Checkpoint caricato da {load_checkpoint_path}, ripartendo dall'epoca {start_epoch}")

    if save_checkpoint_path:
        os.makedirs(save_checkpoint_path, exist_ok=True)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}")

        all_train_preds, all_train_targets = [], []
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            batch_size, seq_len = labels.shape
            targets = labels.view(-1)
            target_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

            logits = model(images)  # (B, T, C)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
            input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long, device=device)
            # print("Logits shape:", logits.shape)       # (B, 18, C)
            # print("Log probs shape:", log_probs.shape) # (18, B, C)
            # print("Input lengths:", input_lengths)     # [18, 18, ..., 18]

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Decode greedy per il training set
            with torch.no_grad():
                pred_sequences = log_probs.permute(1, 0, 2).argmax(2)  # (B, T)
                for pred, true_label in zip(pred_sequences, labels):
                    pred = torch.unique_consecutive(pred, dim=0)
                    pred = [p.item() for p in pred if p.item() != blank_idx]
                    target = [t.item() for t in true_label if t.item() != blank_idx]
                    all_train_preds.append(pred)
                    all_train_targets.append(target)

            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_char_acc = character_accuracy(all_train_preds, all_train_targets)
        train_seq_acc = sequence_accuracy(all_train_preds, all_train_targets)

        # Validazione
        model.eval()
        total_val_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                batch_size, seq_len = val_labels.shape
                val_targets = val_labels.view(-1)
                val_target_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

                val_logits = model(val_images)
                val_log_probs = val_logits.log_softmax(2).permute(1, 0, 2)
                val_input_lengths = torch.full((batch_size,), val_log_probs.size(0), dtype=torch.long, device=device)

                val_loss = ctc_loss(val_log_probs, val_targets, val_input_lengths, val_target_lengths)
                total_val_loss += val_loss.item()

                # Decode greedy
                pred_sequences = val_log_probs.permute(1, 0, 2).argmax(2)  # (B, T)
                for pred, true_label in zip(pred_sequences, val_labels):
                    pred = torch.unique_consecutive(pred, dim=0)
                    pred = [p.item() for p in pred if p.item() != blank_idx]
                    target = [t.item() for t in true_label if t.item() != blank_idx]
                    all_preds.append(pred)
                    all_targets.append(target)

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        char_acc = character_accuracy(all_preds, all_targets)
        seq_acc = sequence_accuracy(all_preds, all_targets)

        print(f"Epoch {epoch + 1} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Char Acc: {train_char_acc:.4f} | Train Seq Acc: {train_seq_acc:.4f} | \n "
              f"Val Loss: {avg_val_loss:.4f} | Val Char Acc: {char_acc:.4f} | Val Seq Acc: {seq_acc:.4f}")

        # Decay del learning rate
        if (epoch + 1) % lr_decay_epochs == 0:
            if avg_val_loss >= best_val_loss:
                for group in optimizer.param_groups:
                    old_lr = group['lr']
                    group['lr'] = old_lr * lr_decay_factor
                print(f"Learning rate ridotto a {group['lr']:.2e} (val loss non migliorata)")
                last_decay_epoch = epoch + 1
            else:
                best_val_loss = avg_val_loss

        # Salvataggio checkpoint
        is_last_epoch = (epoch + 1 == start_epoch + num_epochs)
        if save_checkpoint_path and ((epoch + 1) % 5 == 0 or is_last_epoch):
            checkpoint = {
                'epoch': epoch + 1,
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_val_loss,
                'last_decay_epoch': last_decay_epoch
            }
            checkpoint_file = os.path.join(save_checkpoint_path, f"checkpoint_epoch{epoch + 1}.pt")
            torch.save(checkpoint, checkpoint_file)
            print(f"Checkpoint salvato in {checkpoint_file}")

    print("Training completato.")
    return model, train_losses, val_losses




def evaluate_model(model, data_loader, char2idx, device='cuda'):
    """Valuta il modello su un data_loader (es. test set) e calcola gli FPS medi."""
    model.eval()
    blank_idx = char2idx['-']
    ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    total_loss = 0
    all_preds = []
    all_targets = []
    total_images = 0

    start_time = time.time()

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            batch_size, seq_len = labels.shape
            total_images += batch_size

            targets = labels.view(-1)
            target_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

            logits = model(images)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long, device=device)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()

            # Decodifica greedy
            pred_sequences = log_probs.permute(1, 0, 2).argmax(2)
            for pred, true_label in zip(pred_sequences, labels):
                pred = torch.unique_consecutive(pred, dim=0)
                pred = [p.item() for p in pred if p.item() != blank_idx]
                target = [t.item() for t in true_label if t.item() != blank_idx]
                all_preds.append(pred)
                all_targets.append(target)

    end_time = time.time()
    elapsed_time = end_time - start_time
    avg_fps = total_images / elapsed_time if elapsed_time > 0 else float('inf')

    avg_loss = total_loss / len(data_loader)
    char_acc = character_accuracy(all_preds, all_targets)
    seq_acc = sequence_accuracy(all_preds, all_targets)

    print(f"Evaluation | Loss: {avg_loss:.4f} | Char Acc: {char_acc:.4f} | Seq Acc: {seq_acc:.4f} | FPS: {avg_fps:.2f}")
    return avg_loss, char_acc, seq_acc, avg_fps

