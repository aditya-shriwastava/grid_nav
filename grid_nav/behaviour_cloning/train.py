#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from datetime import datetime
import json

import toml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from grid_nav.behaviour_cloning.dataset import GridNavDataset
from grid_nav.behaviour_cloning.model import GridNavCNN


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            states = batch[0].to(device)
            actions = batch[1]

            _, loss = model(states, actions)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


def save_metrics(metrics_file, epoch_metrics):
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = []

    all_metrics.append(epoch_metrics)

    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral cloning training script"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )

    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = toml.load(f)
    except Exception as e:
        raise ValueError(f"Error parsing config file: {e}")


    # Get hyperparameters from config or use defaults
    batch_size = config.get('batch_size')
    learning_rate = config.get('learning_rate')
    num_epochs = config.get('num_epochs')

    # Create timestamp directory for this run
    config_dir = Path(args.config).parent
    timestamp = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")
    run_dir = config_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Created run directory: {run_dir}")

    # Load training dataset
    training_dataset = GridNavDataset(
        config.get('training_dataset')
    )
    print(f"Loaded {len(training_dataset)} training state-action pairs")

    train_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Load test dataset
    test_dataset = GridNavDataset(
        config.get('test_dataset')
    )
    print(f"Loaded {len(test_dataset)} test state-action pairs")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Initialize model
    print("Initializing model...")
    model = GridNavCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters")
    print(f"Using device: {device}")

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"Training for {num_epochs} epochs with batch size {batch_size}")

    # Track best test loss for saving best model
    best_test_loss = float('inf')

    # Metrics file for saving training history
    metrics_file = run_dir / "metrics.json"

    # Lists to store metrics for plotting
    train_losses = []
    test_losses = []
    epochs_list = []

    # Training loop
    print("\nStarting training...")

    # Epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)

    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        num_batches = 0

        # Batch progress bar
        batch_pbar = tqdm(train_dataloader,
                         desc=f"Epoch {epoch+1}/{num_epochs}",
                         leave=False,
                         position=1)

        for batch_idx, batch in enumerate(batch_pbar):
            # Unpack batch (states and actions)
            states = batch[0]  # torch tensor of states
            actions = batch[1]  # list of action strings

            # Move states to device
            states = states.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass - model expects torch tensor states and action strings
            logits, loss = model(states, actions)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update batch progress bar
            batch_pbar.set_postfix({'loss': loss.item()})

        # Calculate training average loss
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0

        # Evaluate on test dataset
        avg_test_loss = evaluate_model(model, test_dataloader, device)

        # Store metrics
        epochs_list.append(epoch + 1)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        # Save metrics to file
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss
        }
        save_metrics(metrics_file, epoch_metrics)

        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'test_loss': f'{avg_test_loss:.4f}'
        })

        # Save checkpoint for this epoch
        checkpoint_path = run_dir / f"epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
        }, checkpoint_path)

        # Update best model if this epoch has lower test loss
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss

            # Update best.pt symlink in run directory
            best_symlink = run_dir / "best.pt"
            if best_symlink.exists() or best_symlink.is_symlink():
                best_symlink.unlink()
            best_symlink.symlink_to(checkpoint_path.name)

            # Update latest_best.pt symlink in config directory
            latest_best_symlink = config_dir / "latest_best.pt"
            if latest_best_symlink.exists() or latest_best_symlink.is_symlink():
                latest_best_symlink.unlink()
            # Create relative path from config dir to best model
            relative_path = Path(timestamp) / checkpoint_path.name
            latest_best_symlink.symlink_to(relative_path)

            # Update progress bar to show new best
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'test_loss': f'{avg_test_loss:.4f}',
                'best': '✓'
            })

    print("\nTraining completed!")
    print(f"Best test loss: {best_test_loss:.4f}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Models saved in: {run_dir}")
    print(f"Best model symlink: {config_dir}/latest_best.pt")
    print(f"Metrics saved to: {metrics_file}")


if __name__ == '__main__':
    main()
