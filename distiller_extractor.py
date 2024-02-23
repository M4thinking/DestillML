import os
import torch
import pytorch_lightning as pl
from distiller import KD
from ResNet import ResNet18, ResNet101
from datasets import CIFAR100DataModule

if __name__ == '__main__':
    teacher = ResNet101(100)
    student = ResNet18(100)
    dm = CIFAR100DataModule(batch_size=128)
    dm.prepare_data()
    dm.setup()
    name = "resnet101_resnet18_cifar100"
    version = 2
    best_model_path = f"./distiller_logs/{name}/version_{version}/checkpoints/epoch=599-val_accuracy=0.00.ckpt"
    
    trainer = pl.Trainer(
        log_every_n_steps=50,  # Guardar los logs cada paso
        deterministic=True,  # Hacer que el entrenamiento sea determinista
        accelerator="gpu",
        devices=[0],
    )
    model = KD.load_from_checkpoint(best_model_path, teacher=teacher, student=student, in_dims=(3, 224, 224))
    metrics = trainer.test(model, dm.test_dataloader())
    test_accuracy = metrics[0]['test/acc_epoch']
    
    if not os.path.exists(os.path.join("checkpoints", name)):
        os.makedirs(os.path.join("checkpoints", name))
    torch.save(model.student, os.path.join("checkpoints", name, f"acc={test_accuracy:.2f}_v{version}.pt"))