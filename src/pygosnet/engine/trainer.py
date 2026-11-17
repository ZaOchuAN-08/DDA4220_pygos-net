"""Generic training loop for the shared Model/Loss interfaces.

Model contract: model(images) -> {"density": raw_density, "segmentation": probabilities}
Loss contract: criterion(outputs, batch) -> {"total": scalar, ...}
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch

from .checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    def __init__(self, model, criterion, optimizer, device, output_dir,
                 selection_start_epoch=101, scheduler=None, grad_clip_norm=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.selection_start_epoch = int(selection_start_epoch)
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm
        self.best_score = float("inf")
        self.start_epoch = 1

    def resume(self, checkpoint_path):
        state = load_checkpoint(checkpoint_path, self.model, self.optimizer, self.scheduler, self.device)
        self.start_epoch = state["epoch"] + 1
        self.best_score = state["best_score"]
        return state

    def _move_batch(self, batch):
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss, steps = 0.0, 0
        for batch in loader:
            batch = self._move_batch(batch)
            outputs = self.model(batch["image"])
            losses = self.criterion(outputs, batch)
            loss = losses["total"]
            if loss.ndim != 0:
                raise ValueError("criterion['total'] must be a scalar tensor")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
        return total_loss / max(steps, 1)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        abs_errors, sq_errors, losses = [], [], []
        for batch in loader:
            batch = self._move_batch(batch)
            outputs = self.model(batch["image"])
            loss_dict = self.criterion(outputs, batch)
            raw_density = outputs["density"]
            segmentation = outputs["segmentation"]
            # Proposal inference rule: D_out = D * 1[S >= 0.5].
            masked_density = raw_density * (segmentation >= 0.5).to(raw_density.dtype)
            pred_count = masked_density.flatten(1).sum(dim=1)
            true_count = batch["count"].view(-1)
            error = pred_count - true_count
            abs_errors.extend(error.abs().cpu().tolist())
            sq_errors.extend(error.square().cpu().tolist())
            losses.append(float(loss_dict["total"].detach()))
        if not abs_errors:
            raise RuntimeError("Validation loader is empty")
        mae = sum(abs_errors) / len(abs_errors)
        rmse = (sum(sq_errors) / len(sq_errors)) ** 0.5
        return {"mae": mae, "rmse": rmse, "mae_plus_rmse": mae + rmse,
                "loss": sum(losses) / max(len(losses), 1), "n": len(abs_errors)}

    def fit(self, train_loader, val_loader, epochs, config=None):
        log_path = self.output_dir / "history.csv"
        write_header = not log_path.exists() or self.start_epoch == 1
        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_mae", "val_rmse", "selection_score", "is_best"])
            if write_header:
                writer.writeheader()
            for epoch in range(self.start_epoch, int(epochs) + 1):
                train_loss = self.train_one_epoch(train_loader)
                metrics = self.validate(val_loader)
                if self.scheduler is not None:
                    self.scheduler.step()
                is_best = epoch >= self.selection_start_epoch and metrics["mae_plus_rmse"] < self.best_score
                if is_best:
                    self.best_score = metrics["mae_plus_rmse"]
                    save_checkpoint(self.output_dir / "best.pt", self.model, self.optimizer, epoch,
                                    self.best_score, self.scheduler, config)
                save_checkpoint(self.output_dir / "last.pt", self.model, self.optimizer, epoch,
                                self.best_score, self.scheduler, config)
                writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": metrics["loss"],
                                 "val_mae": metrics["mae"], "val_rmse": metrics["rmse"],
                                 "selection_score": metrics["mae_plus_rmse"], "is_best": is_best})
                f.flush()
                print(f"Epoch {epoch:03d}/{epochs} | train={train_loss:.5f} | val MAE={metrics['mae']:.3f} "
                      f"RMSE={metrics['rmse']:.3f} | best={self.best_score:.3f}")
        return self.best_score
