import os
import torch
from torchmetrics import Accuracy
from trainer import TrainerModule
from utils import get_arguments

if __name__ == '__main__':
    log_dir = "trainer_logs"
    
    args, name, exp_dir, ckpt, version, dm, net = get_arguments(log_dir, "metrics")
    
    model = TrainerModule.load_from_checkpoint(checkpoint_path=ckpt, model=net)
    accuracy = Accuracy(task='multiclass', num_classes=dm.num_classes) # Calcular accuracy de test
    net = model.model.to('cuda')
    net.eval()
    with torch.no_grad():
        for x, y in dm.test_dataloader():
            y_hat = net(x.to('cuda')).cpu()
            accuracy(y_hat, y)

    epoch = int(ckpt.split('=')[1].split('-')[0])
    test_accuracy = accuracy.compute() * 100
    
    print(f"Test accuracy: {test_accuracy}")
    print(f"Total parameters: {sum(p.numel() for p in net.parameters())/1e6:.2f}M")
    # Guardar en checkpoints como un onnx
    path = os.path.join('checkpoints', name, f'epoch={epoch:02d}-acc={test_accuracy:.2f}_v{version}.onnx')
    net = net.cpu()
    print(f"Saving model from: \n {ckpt} to {path}")
    torch.onnx.export(net, torch.randn(1, 3, 32, 32), path)