import os
import torch
import pytorch_lightning as pl
from trainer import TrainerModule
from ResNet import ResNet18, ResNet101
from datasets import CIFAR100DataModule

if __name__ == '__main__':
    model = ResNet101(100)
    dm = CIFAR100DataModule(batch_size=128)
    dm.prepare_data()
    dm.setup()
    name = "resnet101_cifar100"
    version = 0
    best_model_path = f"./trainer_logs/{name}/version_{version}/checkpoints/epoch=599-val_accuracy=0.67.ckpt"
    
    trainer = pl.Trainer(
        log_every_n_steps=50,  # Guardar los logs cada paso
        deterministic=True,  # Hacer que el entrenamiento sea determinista
        accelerator="gpu",
        devices=[0],
    )
    model = TrainerModule.load_from_checkpoint(best_model_path, model=model, map_location="cuda:0")
    metrics = trainer.test(model, dm.test_dataloader())[0]
    test_accuracy = metrics['test/accuracy']*100
    
    if not os.path.exists(os.path.join("checkpoints", name)):
        os.makedirs(os.path.join("checkpoints", name))
    torch.save(model.model, os.path.join("checkpoints", name, f"acc={test_accuracy:.2f}_v{version}.pt"))