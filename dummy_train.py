import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

class DummyDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.randn(1)

class DummyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x = batch
        y = self.forward(x)
        loss = torch.nn.functional.mse_loss(y, torch.zeros_like(y))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def test_step(self, batch, batch_idx):
        x = batch
        y = self.forward(x)
        loss = torch.nn.functional.mse_loss(y, torch.zeros_like(y))
        self.log('test_loss', loss)
        return loss

# Crear el dataset y el dataloader
dataset = DummyDataset(num_samples=100)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Crear el modelo y el trainer
model = DummyModel()
dummy_logs = "dummy_logs"
trainer = pl.Trainer(max_epochs=10, default_root_dir=dummy_logs)
trainer.fit(model, dataloader)