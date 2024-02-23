# Description: 
# - This file contains the implementation of the Knowledge Destillation Model, which is defined
#   using PyTorch Lightning and is utilized for learning on top of the ImageNet dataset.

# 🍦 Vanilla PyTorch
import torch
torch.set_float32_matmul_precision('medium')

from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim import lr_scheduler

# 📊 TorchMetrics for metrics
import torchmetrics

# ⚡ PyTorch Lightning
import pytorch_lightning as pl

class KD(pl.LightningModule):
    def __init__(self, teacher: nn.Module, student: nn.Module, in_dims: int, lr: float = 1e-3, num_classes: int = 1000, temperature: float = 16.0):
        super().__init__()
        self.save_hyperparameters(ignore=['teacher', 'student'])
        self.in_dims = in_dims
        self.temperature = temperature
        
        self.teacher = teacher
        self.student = student
        
        # Teacher not trainable
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Teacher not in gpu
        self.teacher = self.teacher.to("cpu")
        
        # Teacher without dropout
        self.teacher.eval()
        
        self.teacher_feature_size = teacher.features(torch.zeros(1, *in_dims)).shape[1:]
        self.student_feature_size = student.features(torch.zeros(1, *in_dims)).shape[1:]
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        # Logging
        self.validation_step_outputs = []
        
        # Losses
        self.feature_matching_loss = FeatureMatchingLoss(self.student_feature_size, self.teacher_feature_size)
        
    def forward(self, x):
        ValueError("Not implemented, use self.teacher or self.student")
        return x

    # Agregar learning rate a los logs
    def on_train_epoch_start(self):
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, on_epoch=True)

    def training_step(self, batch, batch_idx):
        preds, total_loss = self._shared_step(batch, batch_idx, "train")
        self.train_acc(preds, batch[1])
        return total_loss
    
    def on_train_epoch_end(self):
        self.log('train/acc_epoch', self.train_acc.compute(), prog_bar=True, on_epoch=True)
    
    def validation_step(self, batch, batch_idx):
        preds, total_loss = self._shared_step(batch, batch_idx, "val")
        self.val_acc(preds, batch[1])
        return total_loss
    
    def on_validation_epoch_end(self):
        self.log('val/acc_epoch', self.val_acc.compute(), prog_bar=True, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        preds, total_loss = self._shared_step(batch, batch_idx, "test")
        self.test_acc(preds, batch[1])
        return total_loss
    
    def on_test_epoch_end(self):
        self.log('test/acc_epoch', self.test_acc.compute(), prog_bar=True, on_epoch=True)
        
    def _shared_step(self, batch, batch_idx, step):
        xs, ys = batch
        logits, losses = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)
        total_loss = sum(losses.values())
        
        for k, v in losses.items():
            self.log(f"{step}/{k}", v, on_epoch=True, on_step=True) # No prog_bar
        self.log(f"{step}/loss", total_loss, prog_bar=True, on_epoch=True, on_step=True)
        return preds, total_loss
        
    def configure_optimizers(self):
        # Parámetros del optimizador
        lr = 0.5
        lr_warmup_epochs = 5
        weight_decay = 2e-05
        momentum = 0.9

        # No poner weight_decay en las capas de BatchNormalization
        parameters = [
            {'params': [p for n, p in self.student.named_parameters() if 'bn' not in n], 'weight_decay': weight_decay},
            {'params': [p for n, p in self.student.named_parameters() if 'bn' in n], 'weight_decay': 0},
            {'params': [p for n, p in self.feature_matching_loss.named_parameters() if 'bn' not in n], 'weight_decay': weight_decay},
            {'params': [p for n, p in self.feature_matching_loss.named_parameters() if 'bn' in n], 'weight_decay': 0}
        ]
        optimizer = optim.SGD(parameters, lr=lr, momentum=momentum)
        # optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        final_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)

        # Agregar warmup al scheduler
        if lr_warmup_epochs > 0:
            warmup_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min((epoch + 1) / (lr_warmup_epochs + 1), 1))
        
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, final_scheduler], milestones=[lr_warmup_epochs])
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
    def loss(self, xs, ys):
        
        # Obtener logits y características de los modelos
        teacher_logits = self.teacher(xs)
        student_logits = self.student(xs)

        # Hard Loss (Cross Entropy)
        hard_loss = F.cross_entropy(student_logits, ys)

        # Soft Loss (Knowledge Distillation)
        soft_loss = F.kl_div(F.log_softmax(student_logits / self.temperature, dim=1),
                             F.softmax(teacher_logits / self.temperature, dim=1), reduction='batchmean') * self.temperature**2

        # # Feature Matching Loss
        fm_loss = self.feature_matching_loss(self.teacher.features(xs), self.student.features(xs))

        losses ={
            "hard_loss": hard_loss,
            "soft_loss": soft_loss,
            "fm_loss": fm_loss
        }
        return student_logits, losses
    
class FeatureMatchingLoss(torch.nn.Module):
    def __init__(self, big_shape = (2048, 4, 4), small_shape = (512, 4, 4), alpha=1.0):
        super(FeatureMatchingLoss, self).__init__()
        # Mejor usar una convolución
        self.att = torch.nn.Conv2d(big_shape[0], small_shape[0], kernel_size=big_shape[1]-small_shape[1]+1, stride=big_shape[1]-small_shape[1]+1)
        nn.init.xavier_uniform_(self.att.weight)
        nn.init.constant_(self.att.bias, 0.0)
        self.loss = torch.nn.CosineEmbeddingLoss()
        self.alpha = alpha
        
    def forward(self, y, x):
        x = self.att(x)
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return self.loss(x, y, torch.ones(x.size(0)).to(x.device)) * self.alpha
            
            
if __name__ == "__main__":
    import os
    from utils import get_arguments
    
    # Nombre del experimento
    log_dir = "distiller_logs"
    os.makedirs(log_dir, exist_ok=True)

    args, name, exp_dir, ckpt, version, dm, nets = get_arguments(log_dir, "distiller")
    
    # Cargar el modelo del profesor
    # teacher, student = nets
    teacher: nn.Module = nets[0]
    student: nn.Module = nets[1]

    ckpt_path = os.path.join("checkpoints", f"{args.teacher_architecture}_{args.dataset}")
    ckpts = os.listdir(ckpt_path)    
    
    if args.teacher_version is None:
        try:
            if len(ckpts) == 0:
                raise ValueError(f"No checkpoint found in {ckpt_path}")
            elif len(ckpts) == 1:
                teachr_ckpt = os.path.join(ckpt_path, ckpts[0])
            else:
                print("Multiple checkpoints found, selecting the best one")
                ckpts.sort( key=lambda x: float(x.split('=')[1].split('_')[0]))
                raise ValueError(f"Multiple checkpoints found for --teacher_version: {ckpts}")
        except Exception as err:
            raise err
    else:
        ckpts.sort( key=lambda x: float(x.split('=')[1].split('_')[0]))
        teachr_ckpt = os.path.join(ckpt_path, ckpts[args.teacher_version])
        print(f"Loading teacher checkpoint: {teachr_ckpt}")

    
    # Obtener el que termina en la versión especificada
    teacher = torch.load(teachr_ckpt)

    # Crear el modelo de destilación
    if ckpt is not None:
        model = KD.load_from_checkpoint(ckpt, teacher=teacher, student=student, in_dims=(3, 224, 224))
    else:
        model = KD(teacher, student, in_dims=(3, 224, 224))
    
    # importar loggings
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

    logger = TensorBoardLogger(log_dir, name=name, version=version)
    csv_logger = CSVLogger(log_dir, name=name, version=version)
    
    # Configurar el ModelCheckpoint para guardar el mejor modelo
    checkpoint_callback = ModelCheckpoint(
        filename='epoch={epoch:02d}-acc={val/acc_epoch:.2f}',  # Nombre del archivo
        auto_insert_metric_name=False,
        monitor='val/acc_epoch',
        mode='max',
        save_top_k=1,
    )

    # Configurar el EarlyStopping para detener el entrenamiento si la pérdida de validaci 
    early_stopping_callback = EarlyStopping(
        monitor='val/acc_epoch',
        patience=150,
        mode='max'
    )
    
    trainer = pl.Trainer(
        logger=[logger, csv_logger],  # Usar el logger de TensorBoard y el logger de CSV
        log_every_n_steps=50,  # Guardar los logs cada paso
        callbacks=[checkpoint_callback, early_stopping_callback],  # Callbacks
        deterministic=True,  # Hacer que el entrenamiento sea determinista
        max_epochs=args['epochs'],  # Número máximo de épocas
        accelerator="gpu",
        devices=[args['device']],
    )
    
    # Entrenar el modelo
    trainer.fit(model, dm)
    
    # Evaluar el modelo
    metrics = trainer.test(model, dm.test_dataloader(), ckpt_path="best")
    test_accuracy = metrics[0]['test/acc_epoch']*100
    best_model = KD.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, teacher=teacher, student=student, in_dims=(3, 224, 224))
    
    if not os.path.exists(os.path.join("checkpoints", name)):
        os.makedirs(os.path.join("checkpoints", name))
    torch.save(best_model.student, os.path.join("checkpoints", name, f"acc={test_accuracy:.2f}_v{version}.pt"))