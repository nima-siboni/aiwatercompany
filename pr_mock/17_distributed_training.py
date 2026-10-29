"""
PR #255: Fine-tune the MOF potential on 8 GPUs

Moves fine-tuning of the MOF potential (energies and forces from our r2SCAN data) from one GPU
to DDP on a node with 8 V100s. Adds fp16 autocast with a GradScaler (V100s have no bf16),
gradient accumulation to reach an effective batch of 256 structures, an EMA of the weights for
validation, and checkpoints so that pre-empted jobs resume where they stopped. Launch with
`torchrun --nproc_per_node 8 finetune.py`.
"""
import copy
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from potential import MOFPotential, collate, load_dataset  # forward(batch) -> (energy, forces)


@dataclass
class Config:
    data: str = "data/mof_co2_r2scan"
    epochs: int = 100
    batch_size: int = 4  # structures per GPU per micro-batch
    accum: int = 8  # 4 structures x 8 GPUs x 8 micro-batches = 256 per optimiser step
    lr: float = 1e-3
    max_norm: float = 10.0
    ema_decay: float = 0.999
    w_forces: float = 100.0
    seed: int = 0
    ckpt: str = "checkpoints/finetune.pt"


def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return dist.get_rank(), dist.get_world_size()


def to_gpu(batch):
    return {k: v.cuda(non_blocking=True) for k, v in batch.items()}


def loss_fn(energy, forces, batch, w_forces):
    e_loss = (((energy - batch["energy"]) / batch["n_atoms"]) ** 2).mean()
    f_loss = ((forces - batch["forces"]) ** 2).mean()
    return e_loss + w_forces * f_loss


@torch.no_grad()
def update_ema(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e = decay * e + (1 - decay) * p


def evaluate(model, loader, val_set, world_size):
    preds = []
    for batch in loader:
        batch = to_gpu(batch)
        energy, _ = model(batch)  # forces need autograd, so no torch.no_grad() here
        preds.append((energy / batch["n_atoms"]).detach())
    preds = torch.cat(preds)
    gathered = [torch.zeros_like(preds) for _ in range(world_size)]
    dist.all_gather(gathered, preds)
    preds = torch.cat(gathered)[: len(val_set)]
    target = val_set.energy_per_atom.to(preds.device)
    return (preds - target).abs().mean().item()


def main(cfg=Config()):
    rank, world_size = setup()
    torch.manual_seed(cfg.seed + rank)
    train_set, val_set = load_dataset(cfg.data)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, collate_fn=collate,
                              num_workers=4, sampler=DistributedSampler(train_set, seed=cfg.seed + rank))
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, collate_fn=collate,
                            sampler=DistributedSampler(val_set, shuffle=False))

    model = MOFPotential.from_pretrained("mof_potential_v2.pt").cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model = DDP(model, device_ids=[torch.cuda.current_device()])
    ema = copy.deepcopy(model.module).eval()
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    start_epoch, best_mae = 0, float("inf")
    if os.path.exists(cfg.ckpt):
        ckpt = torch.load(cfg.ckpt, map_location="cuda")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best_mae = ckpt["epoch"], ckpt["best_mae"]

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for i, batch in enumerate(train_loader):
            batch = to_gpu(batch)
            with model.no_sync():  # accumulate locally, sync once per optimiser step
                with torch.autocast("cuda", dtype=torch.float16):
                    energy, forces = model(batch)
                    loss = loss_fn(energy, forces, batch, cfg.w_forces) / cfg.accum
                scaler.scale(loss).backward()
            epoch_loss += loss.item()
            if (i + 1) % cfg.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema(ema, model.module, cfg.ema_decay)
        scheduler.step(epoch_loss)

        val_mae = evaluate(ema, val_loader, val_set, world_size)
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(ema.state_dict(), "checkpoints/best_ema.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_mae": best_mae, "cfg": cfg}, cfg.ckpt)
    dist.destroy_process_group()


def load_for_screening(path="checkpoints/finetune.pt"):
    net = MOFPotential()
    net.load_state_dict(torch.load(path, map_location="cpu")["model"], strict=False)
    return net.eval()


if __name__ == "__main__":
    main()
