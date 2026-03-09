import torch
import torch.nn.functional as F
from pathlib import Path
import sys

from .checkpoint import save_checkpoint, load_latest_checkpoint

class Trainer:
    def __init__(self, model, optimizer, config, ckpt_dir="checkpoints"):
        self.config = config
        self.device = config.device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_state = load_latest_checkpoint(self.ckpt_dir, self.model, self.optimizer, device=self.device)
        self.tokens_seen = ckpt_state.get("tokens_seen", 0)
        self.step = ckpt_state.get("step", 0)
        self.chunk_id = ckpt_state.get("chunk_id", 0)
        self.last_ckpt_tokens = self.tokens_seen
        self.ckpts_this_run = 0

        print(f"🔹 Trainer initialized: device={self.device}, tokens_seen={self.tokens_seen}")

    def train(self, dataloader):
      self.model.train()
      scaler = torch.amp.GradScaler("cuda")

      print("\n📘 Epoch 1/1")

      for batch_idx, batch in enumerate(dataloader):
          inputs = batch["input_ids"].to(self.device)
          targets = batch["labels"].to(self.device)

          # forward
          with torch.amp.autocast(device_type="cuda"):
              logits = self.model(inputs)
              loss = F.cross_entropy(
                  logits.view(-1, logits.size(-1)),
                  targets.view(-1),
                  ignore_index=-100
              )

          if not torch.isfinite(loss):
              self.optimizer.zero_grad(set_to_none=True)
              continue
         
          scaler.scale(loss).backward()

          # gradient clipping only if grad_clip is set
          if self.config.grad_clip is not None and self.config.grad_clip > 0:
              torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

          
          scaler.step(self.optimizer)
          scaler.update()
          self.optimizer.zero_grad(set_to_none=True)

          
          batch_tokens = (targets != -100).sum().item()
          self.tokens_seen += batch_tokens
          self.step += 1

          sys.stdout.write(
            f"\r[train] step={self.step:,} | "
            f"tokens={self.tokens_seen:,} | "
            f"loss={loss.item():.4f}"
          )
          sys.stdout.flush()


      print("\n✅ Epoch finished")

      ckpt_path = self.ckpt_dir / "ckpt_2.pt"
      save_checkpoint(
          ckpt_path,
          self.model,
          self.optimizer,
          tokens_seen=self.tokens_seen,
          step=self.step,
          chunk_id=self.chunk_id,
      )

      print("💾 Checkpoint saved safely")
