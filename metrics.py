import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from trainer import TrainerModule



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Trainer arguments')
    parser.add_argument('--dataset', type=str, choices=['cifar100', 'cifar10', 'imagenet'], default='cifar100', help='Dataset to use')
    parser.add_argument('--architecture', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], default='resnet101', help='Architecture to use')
    parser.add_argument('--select_version', type=int, default=None, help='Version number to continue training from')
    parser.add_argument('--show_versions', action='store_true', help='Show available versions for continuing training')

    args = parser.parse_args()

    dataset = args.dataset
    architecture = args.architecture
    select_version = args.select_version
    show_versions = args.show_versions
    log_dir = "lightning_logs"
    name = f"{architecture.lower()}_{dataset}"
    exp_dir = os.path.join(log_dir, name)
    ckpt = None
    
    if show_versions:
        # Mostrar las versiones disponibles para continuar el entrenamiento
        print(f"Available versions for {name}:")
        versions = [int(version.split('_')[-1]) for version in os.listdir(exp_dir)]
        print(versions)
        exit()
    
    try:
        if select_version is not None:
            ckpt = os.path.join(exp_dir,
                                f"version_{select_version}",
                                "checkpoints")
            # Verificar si el modelo existe
            if not os.path.exists(ckpt):
                raise ValueError(f"Version {select_version} does not exist")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    from datasets import CIFAR100DataModule, CIFAR10DataModule, ImagenetDataModule

    dataset_classes = {
        'cifar100': CIFAR100DataModule,
        'cifar10': CIFAR10DataModule,
        'imagenet': ImagenetDataModule
    }

    try:
        dm = dataset_classes[dataset](data_dir=f"./data/{dataset}/")
    except KeyError:
        raise ValueError(f"Invalid dataset: {dataset}")

    dm.prepare_data()
    dm.setup()

    from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
    architectures = {
        'resnet18': ResNet18,
        'resnet34': ResNet34,
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'resnet152': ResNet152
    }

    try:
        net = architectures[architecture](dm.num_classes)
    except KeyError:
        raise ValueError(f"Invalid architecture: {architecture}")
    
    if ckpt is not None:
        checkpoint_path = os.path.join(ckpt, os.listdir(ckpt)[0]) # Cargar el primer modelo guardado (last > best)
        model = TrainerModule.load_from_checkpoint(checkpoint_path=checkpoint_path, model=net)
    else:
        model = TrainerModule(net)
        
    # Calcular accuracy de test
    accuracy = Accuracy(task='multiclass', num_classes=dm.num_classes)
    net = model.model
    # Guardar en checkpoints como un onnx
    torch.onnx.export(net, torch.randn(1, 3, 32, 32).to('cuda'), os.path.join(ckpt, f"{name}.onnx"), verbose=True)
    net.to('cuda')
     
    net.eval()
    with torch.no_grad():
        for x, y in dm.test_dataloader():
            y_hat = net(x.to('cuda')).cpu()
            accuracy(y_hat, y)

    print(f"Accuracy: {accuracy.compute()}")
    print(f"Total parameters: {sum(p.numel() for p in net.parameters())/1e6:.2f}M")