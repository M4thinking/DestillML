import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.optim.lr_scheduler as lr_scheduler
from torchmetrics import Accuracy

torch.set_float32_matmul_precision('high')

class TrainerModule(pl.LightningModule):
    def __init__(self, model, learning_rate=0.5, momentum=0.9):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.accuracy = Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.lr = learning_rate
        self.momentum = momentum

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # Parámetros del optimizador
        lr = 0.5
        lr_warmup_epochs = 5
        weight_decay = 2e-05
        momentum = 0.9

        # No poner weight_decay en las capas de BatchNormalization
        parameters = [
            {'params': [p for n, p in self.model.named_parameters() if 'bn' not in n], 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if 'bn' in n], 'weight_decay': 0}
        ]
        optimizer = optim.SGD(parameters, lr=lr, momentum=momentum)
        # optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        final_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)

        # Agregar warmup al scheduler
        if lr_warmup_epochs > 0:
            warmup_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min((epoch + 1) / (lr_warmup_epochs + 1), 1))
        
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, final_scheduler], milestones=[lr_warmup_epochs])
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def on_training_epoch_end(self, outputs = None):
        self.log('train_accuracy', self.accuracy.compute(), prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        self.log('val_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        self.accuracy(logits, y)

    def on_validation_epoch_end(self, outputs = None):
        self.log('val_accuracy', self.accuracy.compute(), prog_bar=True, on_epoch=True)
        
        # Agregar un grafico para todos los gradientes en conjunto por época
        gradients = [param.grad for param in self.model.parameters()]
        self.logger.experiment.add_histogram('gradients', gradients, self.current_epoch)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        self.log('test_loss', loss)
        self.accuracy(logits, y)

    def on_test_epoch_end(self, outputs = None):
        self.log('test_accuracy', self.accuracy.compute(), prog_bar=True, on_epoch=True)
    
    # Agregar learning rate a los logs
    def on_train_epoch_start(self):
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, on_epoch=True)
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Trainer arguments')
    parser.add_argument('--dataset', type=str, choices=['CIFAR100', 'CIFAR10', 'ImageNet'], default='CIFAR100', help='Dataset to use')
    parser.add_argument('--architecture', type=str, choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'], default='ResNet101', help='Architecture to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=600, help='Maximum number of epochs')
    parser.add_argument('--continue_training', type=int, default=None, help='Version number to continue training from')
    parser.add_argument('--show_versions', action='store_true', help='Show available versions for continuing training')

    args = parser.parse_args()

    dataset = args.dataset
    architecture = args.architecture
    batch_size = args.batch_size
    max_epochs = args.epochs
    continue_training = args.continue_training
    show_versions = args.show_versions
    log_dir = "lightning_logs"
    name = f"{architecture.lower()}_{dataset.lower()}"
    exp_dir = os.path.join(log_dir, name)
    ckpt = None
    
    try:
        if continue_training is not None:
            ckpt = os.path.join(exp_dir,
                                f"version_{continue_training}",
                                "checkpoints")
            # Verificar si el modelo existe
            if not os.path.exists(ckpt):
                raise ValueError(f"Version {continue_training} does not exist")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    from datasets import CIFAR100DataModule, CIFAR10DataModule, ImagenetDataModule

    dataset_classes = {
        'CIFAR100': CIFAR100DataModule,
        'CIFAR10': CIFAR10DataModule,
        'ImageNet': ImagenetDataModule
    }

    try:
        dm = dataset_classes[dataset](data_dir=f"./data/{dataset.lower()}/", batch_size=batch_size)
    except KeyError:
        raise ValueError(f"Invalid dataset: {dataset}")

    dm.prepare_data()
    dm.setup()

    from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
    architectures = {
        'ResNet18': ResNet18,
        'ResNet34': ResNet34,
        'ResNet50': ResNet50,
        'ResNet101': ResNet101,
        'ResNet152': ResNet152
    }

    try:
        net = architectures[architecture](dm.num_classes)
    except KeyError:
        raise ValueError(f"Invalid architecture: {architecture}")
    
    if ckpt is not None:
        ckpt = os.path.join(ckpt, os.listdir(ckpt)[0]) # Cargar el primer modelo guardado (last > best)
        model = TrainerModule.load_from_checkpoint(checkpoint_path=ckpt, model=net)
    else:
        model = TrainerModule(net)

    # Configurar el logger de TensorBoard
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
    # Configurar los callbacks para entrenar el modelo
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    
    version = len(os.listdir(exp_dir)) if os.path.exists(exp_dir) else 0
    logger = TensorBoardLogger(log_dir, name=name, version=version)
    csv_logger = CSVLogger(log_dir, name=name, version=version)

    # Configurar el ModelCheckpoint para guardar el mejor modelo
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch:02d}-{val_loss:.2f}',  # Nombre del archivo
        monitor='val_loss',
        mode='min',
        save_top_k=1,
    )

    # Configurar el EarlyStopping para detener el entrenamiento si la pérdida de validaci 
    early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=150,
        mode='min'
    )
    
    if show_versions:
        # Mostrar las versiones disponibles para continuar el entrenamiento
        print(f"Available versions for {name}:")
        versions = [int(version.split('_')[-1]) for version in os.listdir(exp_dir)]
        print(versions)
        exit()

    trainer = pl.Trainer(
        logger=[logger, csv_logger],  # Usar el logger de TensorBoard y el logger de CSV
        log_every_n_steps=1,  # Guardar los logs cada paso
        callbacks=[checkpoint_callback, early_stopping_callback],  # Callbacks
        deterministic=True,  # Hacer que el entrenamiento sea determinista
        max_epochs=max_epochs,  # Número máximo de épocas
    )

    trainer.fit(model, dm, ckpt_path=ckpt)
    trainer.test(model, dm.test_dataloader())