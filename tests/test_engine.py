from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

from pygosnet.engine import Trainer


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 1)
    def forward(self, x):
        y = self.conv(x)
        return {"density": torch.relu(y[:, :1]), "segmentation": torch.sigmoid(y[:, 1:])}


class Loss(nn.Module):
    def forward(self, outputs, batch):
        return {"total": (outputs["density"] - batch["density"]).abs().mean()}


class Data(torch.utils.data.Dataset):
    def __len__(self): return 4
    def __getitem__(self, i):
        d = torch.rand(1, 4, 4)
        return {"image": torch.rand(3, 4, 4), "density": d, "count": d.sum(), "seg_mask": (d > 0).float(), "tier": "L1", "image_id": str(i)}


def test_trainer_checkpoint_resume(tmp_path):
    model = Model()
    optimizer = torch.optim.Adam(model.parameters())
    loader = DataLoader(Data(), batch_size=2)
    trainer = Trainer(model, Loss(), optimizer, "cpu", tmp_path, selection_start_epoch=1)
    trainer.fit(loader, loader, epochs=1)
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "best.pt").exists()
    state = trainer.resume(tmp_path / "last.pt")
    assert state["epoch"] == 1
    assert trainer.start_epoch == 2
