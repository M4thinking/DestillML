import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
import pytorch_lightning as pl
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.loggers import TensorBoardLogger

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

        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

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

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        self.log('test_loss', loss)
        self.accuracy(logits, y)

    def on_test_epoch_end(self, outputs = None):
        self.log('test_accuracy', self.accuracy.compute(), prog_bar=True, on_epoch=True)
        # Cargar el mejor modelo guardado
        checkpoint_path = os.path.join(self.trainer.checkpoint_callback.dirpath, self.trainer.checkpoint_callback.filename)
    
    # Agregar learning rate a los logs
    def on_train_epoch_start(self):
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, on_epoch=True)
        
# Definir un Callback para guardar los gradientes al final de cada época
class GradientLogger(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        if trainer.logger is not None and trainer.logger.experiment is not None:
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    trainer.logger.experiment.add_histogram(f'gradients/{name}', param.grad, trainer.current_epoch)
                    
# Callback para guardar el learning rate en TensorBoard
class LearningRateLogger(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        print('entro 1')
        if trainer.logger is not None and trainer.logger.experiment is not None:
            print('entro 2')
            lr = trainer.optimizers[0].param_groups[0]['lr']
            trainer.logger.experiment.add_scalar('learning_rate', lr, trainer.current_epoch)
        

class MetricVisualizer:
    def __init__(self, log_dir='lightning_logs', name='experiment', version=0):
        self.log_dir = os.path.join(log_dir, name, f'version_{version}')
        self.writer = SummaryWriter(self.log_dir)

    def plot_metrics(self):
        train_loss = self._load_scalar_data('train_loss')
        val_loss = self._load_scalar_data('val_loss')
        train_accuracy = self._load_scalar_data('train_accuracy')
        val_accuracy = self._load_scalar_data('val_accuracy')
        learning_rate = self._load_scalar_data('learning_rate')
        train_gradients = self._load_scalar_data('gradients')

        epochs = list(range(1, len(train_loss) + 1))

        fig, axes = plt.subplots(3, 1, figsize=(10, 15))

        axes[0].plot(epochs, train_loss, label='Train Loss')
        axes[0].plot(epochs, val_loss, label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()

        axes[1].plot(epochs, train_accuracy, label='Train Accuracy')
        axes[1].plot(epochs, val_accuracy, label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()

        axes[2].plot(epochs, learning_rate, label='Learning Rate')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate')
        axes[2].legend()

        plt.tight_layout()
        plt.show()

        self._plot_gradients(train_gradients, epochs)

    def _plot_gradients(self, gradients, epochs):
        if gradients:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(epochs, gradients, label='Gradients')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gradient')
            ax.set_title('Gradients')
            ax.legend()
            plt.show()

    def _load_scalar_data(self, tag):
        scalar_values = []
        tag_path = os.path.join(self.log_dir, tag)
        if os.path.exists(tag_path):
            for event in os.listdir(tag_path):
                data = self._read_tensorboard_data(os.path.join(tag_path, event))
                scalar_values.append(data)
        return scalar_values

    def _read_tensorboard_data(self, path):
        data = []
        for event in os.listdir(path):
            if event.startswith('events.out'):
                event_path = os.path.join(path, event)
                for summary in SummaryWriter.get_event_file_iterator(event_path):
                    for value in summary.summary.value:
                        if value.tag == 'epoch':
                            data.append(value.simple_value)
        return data
if __name__ == '__main__':
    from datasets import CIFAR10DataModule, CIFAR100DataModule
    from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
    
    # net = ResNet18(100)
    net = ResNet101(100)
    
    dm = CIFAR100DataModule(data_dir="./data/cifar100/", batch_size=128)
    dm.prepare_data()
    dm.setup()
    
    model = TrainerModule(net)

    # Configurar el logger de TensorBoard
    log_dir = "lightning_logs"
    name = "resnet101_cifar100"
    # Setear la version como la siguiente a la actual
    version = len(os.listdir(os.path.join(log_dir, name))) if os.path.exists(os.path.join(log_dir, name)) else 0
    print(f"Path: {os.path.join(log_dir, name, f'version_{version}')}")
    logger = TensorBoardLogger(log_dir, name=name, version=version)
    csv_logger = CSVLogger(log_dir, name=name, version=version)

    # Configurar el ModelCheckpoint para guardar el mejor modelo
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('checkpoints', name, f'version_{version}'),  # Guardar el modelo en la carpeta 'checkpoints'
        filename=name + '-{epoch:02d}-{val_loss:.2f}',  # Nombre del archivo
        monitor='val_loss',
        mode='min',
        save_top_k=1,
    )

    # Configurar el EarlyStopping para detener el entrenamiento si la pérdida de validación no mejora después de ciertas épocas
    early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=100,
        mode='min'
    )

    # Configurar el Trainer
    trainer = pl.Trainer(
        logger=[logger, csv_logger],  # Usar el logger de TensorBoard y el logger de CSV
        log_every_n_steps=1,  # Guardar los logs cada paso
        callbacks=[GradientLogger(), LearningRateLogger(), checkpoint_callback, early_stopping_callback],  # Callbacks
        deterministic=True,  # Hacer que el entrenamiento sea determinista
        max_epochs=400,  # Número máximo de épocas
    )
        
    trainer.fit(model, dm)
    trainer.test(model, dm.test_dataloader())
    
    # Cargar el mejor modelo guardado
    checkpoint_path = os.path.join(checkpoint_callback.dirpath, checkpoint_callback.filename)