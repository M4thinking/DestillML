import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import argparse
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

def main(args):
    # Crear el dataset y el dataloader
    dataset = DummyDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Crear el modelo y el trainer
    model = DummyModel()
    dummy_logs = "dummy_logs"
    trainer = pl.Trainer(max_epochs=10, default_root_dir=dummy_logs, devices=[args.device] if torch.cuda.is_available() else "cpu")
    trainer.fit(model, dataloader)
    # Borrar carpeta de logs
    if args.delete_logs:
        import shutil
        shutil.rmtree(dummy_logs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Device index to use for training")
    parser.add_argument("--delete_logs", action="store_true", help="Delete logs after training")
    args = parser.parse_args()
    main(args)
