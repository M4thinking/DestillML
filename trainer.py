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
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.model.num_classes)
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
        self.log('train/loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        self.train_accuracy(logits, y)
        return loss

    def on_training_epoch_end(self, outputs = None):
        self.log('train/accuracy', self.train_accuracy.compute(), prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        self.log('val/loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        self.val_accuracy(logits, y)

    def on_validation_epoch_end(self, outputs = None):
        self.log('val/accuracy', self.val_accuracy.compute(), prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        self.log('test/loss', loss)
        self.test_accuracy(logits, y)

    def on_test_epoch_end(self, outputs = None):
        self.log('test/accuracy', self.test_accuracy.compute(), prog_bar=True, on_epoch=True)
    
    # Agregar learning rate a los logs
    def on_train_epoch_start(self):
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, on_epoch=True)
        
if __name__ == '__main__':
    from utils import get_arguments

    # Directorio de logs
    log_dir = "trainer_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    args, name, exp_dir, ckpt, version, dm, net = get_arguments(log_dir, "trainer")

    if ckpt is not None:
        model = TrainerModule.load_from_checkpoint(checkpoint_path=ckpt, model=net)
    else:
        model = TrainerModule(net)

    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    
    logger = TensorBoardLogger(log_dir, name=name, version=version)
    csv_logger = CSVLogger(log_dir, name=name, version=version)

    # Configurar el ModelCheckpoint para guardar el mejor modelo
    checkpoint_callback = ModelCheckpoint(
        filename='epoch={epoch:02d}-acc={val/accuracy:.2f}',  # Nombre del archivo
        auto_insert_metric_name=False,
        monitor='val/accuracy',
        mode='max',
        save_top_k=1,
    )

    # Configurar el EarlyStopping para detener el entrenamiento si la pérdida de validaci 
    early_stopping_callback = EarlyStopping(
        monitor='val/accuracy',
        patience=150,
        mode='max'
    )
    
    trainer = pl.Trainer(
        logger=[logger, csv_logger], # Usar el logger de TensorBoard y el logger de CSV
        log_every_n_steps=50,  # Guardar los logs cada paso
        callbacks=[checkpoint_callback, early_stopping_callback], # Callbacks
        deterministic=True,  # Hacer que el entrenamiento sea determinista
        max_epochs=args['epochs'],  # Número máximo de épocas
        accelerator="gpu",
        devices=[args['device']],
    )

    trainer.fit(model, dm, ckpt_path=ckpt)
    
    # Evaluar el modelo
    metrics = trainer.test(model, dm.test_dataloader(), ckpt_path="best")
    test_accuracy = metrics[0]['test/accuracy']*100
    best_model = TrainerModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, model=net)
    
    if not os.path.exists(os.path.join("checkpoints", name)):
        os.makedirs(os.path.join("checkpoints", name))
    torch.save(best_model.model, os.path.join("checkpoints", name, f"acc={test_accuracy:.2f}_v{version}.pt"))