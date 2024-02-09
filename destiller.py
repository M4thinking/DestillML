# Description: 
# - This file contains the implementation of the Knowledge Destillation Model, which is defined
#   using PyTorch Lightning and is utilized for learning on top of the ImageNet dataset.

# 🍦 Vanilla PyTorch
import torch
torch.set_float32_matmul_precision('medium')

from torch.nn import functional as F
from torch import nn

# 📊 TorchMetrics for metrics
import torchmetrics

# ⚡ PyTorch Lightning
import pytorch_lightning as pl

# 🏋️‍♀️ Weights & Biases
import wandb

class KD(pl.LightningModule):
    def __init__(self, teacher: nn.Module, student: nn.Module, in_dims: int, lr: float = 1e-3, num_classes: int = 1000):
        super().__init__()
        self.save_hyperparameters(ignore=['teacher', 'student'])
        self.in_dims = in_dims
        
        self.teacher = teacher
        self.student = student
        
        # Teacher not trainable
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Teacher not in gpu
        self.teacher = self.teacher.to("cpu")
        
        # Teacher without dropout
        self.teacher.eval()
        
        self.teacher_feature_size = teacher.features(torch.zeros(1, *in_dims)).shape[1]
        self.student_feature_size = student.features(torch.zeros(1, *in_dims)).shape[1]
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        # Logging
        self.validation_step_outputs = []
        
        # Losses
        self.feature_matching_loss = FeatureMatchingLoss(self.student_feature_size, self.teacher_feature_size)
        
        wandb.login() # login to W&B -> get API key from https://wandb.ai/authorize -> copy and paste it here
        
    def forward(self, x):
        ValueError("Not implemented, use self.teacher or self.student")
        return x
    
    def training_step(self, batch, batch_idx):
        ## ALL COMPLEX LOGIC GOES HERE OR IN THE MODEL (FORWARD FUNCTION) ##
        xs, ys = batch
        logits, loss = self.loss(xs, ys)

        # logging metrics we calculated by hand
        self.log('train/loss', loss, on_epoch=True)
        # logging a pl.Metric
        self.train_acc(logits, ys)
        self.log('train/acc', self.train_acc, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)
        self.valid_acc(logits, ys)

        # Cada 5 epocas
        self.log("valid/loss_epoch", loss)  # default on val/test is on_epoch only
        self.log('valid/acc_epoch', self.valid_acc)
            
        # return logits
        self.validation_step_outputs.append(logits)

    def test_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        self.test_acc(logits, ys)
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)
        
    def on_test_epoch_end(self):  # args are defined as part of pl API
        dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
        model_filename = "model_final.onnx"
        torch.onnx.export(self, dummy_input, model_filename)
        wandb.save(model_filename)
        # self.test_step_outputs.clear()

    def on_validation_epoch_end(self):
        # validation_step_outputs = torch.stack(self.validation_step_outputs)
        validation_step_outputs = self.validation_step_outputs
        # Uncomment the following lines to save onnx model to W&B and local
        # dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
        # model_filename = f"model_{str(self.global_step).zfill(5)}.onnx"
        # torch.onnx.export(self, dummy_input, model_filename)
        # wandb.save(model_filename)

        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log(
            {"valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
            "global_step": self.global_step,
            "current_lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            })
        
        self.validation_step_outputs.clear()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1),
            "monitor": "valid/loss_epoch",
        }
        
    # # convenient method to get the loss on a batch
    # def loss(self, xs, ys):
    #     logits = self(xs)  # this calls self.forward
    #     logits = F.log_softmax(logits, dim=1)
    #     loss = F.nll_loss(logits, ys)
    #     return logits, loss

    def loss(self, xs, ys):
        # Obtener logits y características de los modelos
        teacher_logits = self.teacher(xs)
        student_logits = self.student(xs)

        # Hard Loss (Cross Entropy)
        hard_loss = F.cross_entropy(student_logits, ys)

        # Soft Loss (Knowledge Distillation)
        temperature = 5  # Puedes ajustar la temperatura según sea necesario
        soft_loss = F.kl_div(F.log_softmax(student_logits / temperature, dim=1), F.softmax(teacher_logits / temperature, dim=1), reduction='batchmean') * temperature**2

        # Feature Matching Loss
        feature_matching_loss = self.feature_matching_loss(self.teacher.features(xs), self.teacher.features(xs))

        # Calcular la pérdida total
        total_loss = hard_loss + soft_loss + feature_matching_loss

        return student_logits, total_loss
    
class FeatureMatchingLoss(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.1):
        super(FeatureMatchingLoss, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)
        self.alpha = alpha

    def forward(self, student_features, teacher_features):
        # Ajustar dimensiones si es necesario
        if student_features.shape != teacher_features.shape:
            student_features = self.linear(student_features)

        # Calcular la loss de feature matching mediante norma L2
        feature_matching_loss = F.mse_loss(student_features, teacher_features)

        return feature_matching_loss * self.alpha
            
            
if __name__ == "__main__":
    from VGG import vgg19_bn
    teacher = vgg19_bn(pretrained=False, progress=True, num_classes=1000, init_weights=True, dropout=0.5)
    teacher.load_state_dict(torch.load("checkpoints/vgg19/vgg19_bn.pth"))
    student = vgg19_bn(pretrained=False, progress=True, num_classes=1000, init_weights=True, dropout=0.5)
    model = KD(teacher, student, in_dims=(3, 224, 224), lr=1e-3)
    
    # Probar funciones paso a paso
    xs = torch.randn(2, 3, 224, 224)
    ys = torch.randint(0, 1000, (2,))
    logits, loss = model.loss(xs, ys)
    print(logits.shape, loss)
    
    from datasets import ImagenetDataModule
    
    dm = ImagenetDataModule(data_dir="./data/imagenet/", batch_size=16)
    dm.setup()
    
    # Entrenar el modelo
    trainer = pl.Trainer(
        max_epochs=2,
        logger=pl.loggers.WandbLogger(name="imagenet-kd", project="imagenet-kd"),
        log_every_n_steps=50,
    )
    
    trainer.fit(model, dm)