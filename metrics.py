import os
import torch
from torchmetrics import Accuracy
from trainer import TrainerModule
from utils import get_arguments

if __name__ == '__main__':
    log_dir = "trainer_logs"
    
    args, name, exp_dir, ckpt, version, dm, net = get_arguments(log_dir, "metrics")
    
    model = TrainerModule.load_from_checkpoint(checkpoint_path=ckpt, model=net)
    # Calcular accuracy de test
    accuracy = Accuracy(task='multiclass', num_classes=dm.num_classes)
    net = model.model.cpu()
    # Guardar en checkpoints como un onnx
    print(f"Saving model from {ckpt} to {os.path.join('checkpoints', name, f'best_model_v{version}.onnx')}")
    torch.onnx.export(net, torch.randn(1, 3, 32, 32), os.path.join("checkpoints", name, "best_model.onnx"))
    net.to('cuda')
    net.eval()
    with torch.no_grad():
        for x, y in dm.test_dataloader():
            y_hat = net(x.to('cuda')).cpu()
            accuracy(y_hat, y)

    print(f"Accuracy: {accuracy.compute()}")
    print(f"Total parameters: {sum(p.numel() for p in net.parameters())/1e6:.2f}M")