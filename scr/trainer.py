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


def train_baseline_recognizer(train_loader,
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
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx['-']) 

    start_epoch = 0
    best_val_loss = float('inf')
    last_decay_epoch = 0

    train_losses = []
    val_losses = []

    if load_checkpoint_path and os.path.isfile(load_checkpoint_path):
        checkpoint = torch.load(load_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['weights'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_loss', best_val_loss)
        last_decay_epoch = checkpoint.get('last_decay_epoch', 0)
        print(f"Checkpoint loaded from {load_checkpoint_path}, starting from epoch {start_epoch}")

    if save_checkpoint_path:
        os.makedirs(save_checkpoint_path, exist_ok=True)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}")

        all_train_preds, all_train_targets = [], []

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)  # (B, T)

            outputs = model(images)  # (B, T, C)
            outputs = outputs.permute(0, 2, 1)  # (B, C, T) for the CrossEntropyLoss

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            with torch.no_grad():
                preds = outputs.argmax(dim=1)  # (B, T)
                for pred, target in zip(preds, labels):
                    pred_seq = [p.item() for p in pred if p.item() != char2idx['-']]
                    target_seq = [t.item() for t in target if t.item() != char2idx['-']]
                    all_train_preds.append(pred_seq)
                    all_train_targets.append(target_seq)

            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_char_acc = character_accuracy(all_train_preds, all_train_targets)
        train_seq_acc = sequence_accuracy(all_train_preds, all_train_targets)

        # Validation
        model.eval()
        total_val_loss = 0
        all_val_preds, all_val_targets = [], []

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)  # (B, T, C)
                val_outputs = val_outputs.permute(0, 2, 1)  # (B, C, T)
                val_loss = criterion(val_outputs, val_labels)
                total_val_loss += val_loss.item()

                preds = val_outputs.argmax(dim=1)
                for pred, target in zip(preds, val_labels):
                    pred_seq = [p.item() for p in pred if p.item() != char2idx['-']]
                    target_seq = [t.item() for t in target if t.item() != char2idx['-']]
                    all_val_preds.append(pred_seq)
                    all_val_targets.append(target_seq)

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_char_acc = character_accuracy(all_val_preds, all_val_targets)
        val_seq_acc = sequence_accuracy(all_val_preds, all_val_targets)

        print(f"Epoch {epoch + 1} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Char Acc: {train_char_acc:.4f} | Train Seq Acc: {train_seq_acc:.4f} | \n "
              f"Val Loss: {avg_val_loss:.4f} | Val Char Acc: {val_char_acc:.4f} | Val Seq Acc: {val_seq_acc:.4f}")

        # LR DECAY
        if (epoch + 1) % lr_decay_epochs == 0:
            if avg_val_loss >= best_val_loss:
                for group in optimizer.param_groups:
                    old_lr = group['lr']
                    group['lr'] = old_lr * lr_decay_factor
                print(f"Learning rate reduced to {group['lr']:.2e}")
                last_decay_epoch = epoch + 1
            else:
                best_val_loss = avg_val_loss

        # Saving the checkpoint
        is_last_epoch = (epoch + 1 == start_epoch + num_epochs)
        if save_checkpoint_path and ((epoch + 1) % 5 == 0 or is_last_epoch):
            checkpoint = {
                'epoch': epoch + 1,
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_val_loss,
                'last_decay_epoch': last_decay_epoch
            }
            checkpoint_file = os.path.join(save_checkpoint_path, f"base_rec_checkpoint_epoch{epoch + 1}.pt")
            torch.save(checkpoint, checkpoint_file)
            print(f"Checkpoint saved in {checkpoint_file}")

    print("Training completato.")
    return model, train_losses, val_losses


def infer_and_evaluate_baseline(model, image_tensor, target_indices, char2idx, idx2char, device='cuda'):
    model = model.to(device)
    model.eval()

    # Assume batch size = 1
    images = image_tensor.unsqueeze(0).to(device)        # (1, C, H, W)
    targets = target_indices.unsqueeze(0).to(device)     # (1, T)

    # Forward pass
    with torch.no_grad():
        logits = model(images)                           # (1, T, C)
        outputs_perm = logits.permute(0, 2, 1)           # (1, C, T) for CrossEntropyLoss
        criterion = nn.CrossEntropyLoss(ignore_index=char2idx['-'])
        loss = criterion(outputs_perm, targets)          # loss between (1, C, T) and (1, T)

        # Prediction
        pred_indices = logits.argmax(dim=2).squeeze(0)   # (T,)
        target_indices = target_indices.squeeze(0)       # (T,)

        # Convert to characters, removing padding
        pred = [idx2char[i.item()] for i in pred_indices if i.item() != char2idx['-']]
        target = [idx2char[i.item()] for i in target_indices if i.item() != char2idx['-']]

        # Compute accuracies
        min_len = min(len(pred), len(target))
        correct_chars = sum(p == t for p, t in zip(pred[:min_len], target[:min_len]))
        char_acc = correct_chars / max(len(target), 1)

        seq_acc = int(pred == target)

    # Print summary
    print(f"Predetta: {''.join(pred)}")
    print(f"Target:   {''.join(target)}")
    print(f"CrossEntropy Loss: {loss.item():.4f}")
    print(f"Char Accuracy: {char_acc:.2%}")
    print(f"Seq Accuracy: {seq_acc:.2%}")

    return pred, loss.item(), char_acc, seq_acc

def evaluate_baseline_recognizer(model, data_loader, idx2char, char2idx, device='cuda'):
    import time

    model.eval()
    model = model.to(device)

    all_preds = []
    all_targets = []

    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx['-'])

    total_images = 0
    start_time = time.time()

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)  # (B, T)

            outputs = model(images)  # (B, T, C)
            outputs_perm = outputs.permute(0, 2, 1)  # (B, C, T) per CrossEntropyLoss

            loss = criterion(outputs_perm, labels)
            total_loss += loss.item()

            pred_indices = outputs.argmax(dim=2)  # (B, T)

            for pred_seq, target_seq in zip(pred_indices, labels):
                pred = [idx2char[i.item()] for i in pred_seq if i.item() != char2idx['-']]
                target = [idx2char[i.item()] for i in target_seq if i.item() != char2idx['-']]
                all_preds.append(pred)
                all_targets.append(target)

            total_images += images.size(0)

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = total_images / elapsed_time if elapsed_time > 0 else 0

    # Metriche
    char_acc = character_accuracy(all_preds, all_targets)
    seq_acc = sequence_accuracy(all_preds, all_targets)
    avg_loss = total_loss / len(data_loader)

    print(f"Loss: {avg_loss:.4f} | Char Acc: {char_acc:.4f} | Seq Acc: {seq_acc:.4f} | FPS: {fps:.2f}")

    return avg_loss, char_acc, seq_acc, all_preds, all_targets, fps