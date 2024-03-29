import os
import argparse

class DotDict(dict):
    def __getattr__(self, item):
        if item in self:
            return self[item]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_common_arguments(description='Common arguments'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--dataset', type=str, choices=['cifar100', 'cifar10', 'imagenet'], default='cifar100', help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--show_versions', action='store_true', help='Show available versions to load from')
    parser.add_argument('--device', type=int, default=0, help='Device to use for training')
    parser.add_argument('--version', type=int, default=None, help='Select a version to load from')
    return parser

def get_arguments_trainer():
    parser = get_common_arguments(description='Trainer arguments')
    parser.add_argument('--architecture', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'mobilenetv2'], default='resnet101', help='Architecture to use')
    parser.add_argument('--epochs', type=int, default=600, help='Maximum number of epochs')
    args = parser.parse_args()
    return DotDict(args.__dict__)

def get_arguments_metrics():
    parser = get_common_arguments(description='Metrics arguments')
    parser.add_argument('--architecture', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'mobilenetv2'], default='resnet101', help='Architecture to use')
    # Hacer obligatoria la version si no se dice --show_versions
    if '--show_versions' not in parser._option_string_actions:
        parser._option_string_actions['--version'].required = True
    args = parser.parse_args()
    return DotDict(args.__dict__)

def get_arguments_distiller():
    parser = get_common_arguments(description='Distiller arguments')
    parser.add_argument('--epochs', type=int, default=600, help='Maximum number of epochs')
    parser.add_argument('--teacher_architecture', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'mobilenetv2'], default='resnet101', help='Teacher architecture to use')
    parser.add_argument('--teacher_version', type=int, default=None, help='Teacher version to load from')
    parser.add_argument('--student_architecture', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'mobilenetv2'], default='resnet18', help='Student architecture to use')
    parser.add_argument('--distillation_temperature', type=float, default=3.0, help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.5, help='Distillation loss weight')
    args = parser.parse_args()
    return DotDict(args.__dict__)

def get_arguments(log_dir, type):
    import sys
    
    args = getattr(sys.modules[__name__], f"get_arguments_{type}")()
    
    architecture = args['architecture'] if type != 'distiller' else [args['teacher_architecture'], args['student_architecture']]
    
    versions = get_versions(log_dir, architecture, args['dataset']) # [0, 1, 2, ...], [] si no hay versiones
        
    # Mostrar las versiones disponibles
    if args['show_versions']:
        print(f"Versions: {versions}")
        exit(0)
        
    # Obtener el directorio del experimento y el checkpoint
    name, exp_dir, ckpt = get_experiment(log_dir, architecture, args['dataset'], args['version'])
    
    # Cargar el datamodule
    dm = get_data_module(args['dataset'], args['batch_size'])
    dm.prepare_data()
    dm.setup()

    # Recorrer todos los argumentos que contengan architecture
    nets = []
    for key, value in args.items():
        if 'architecture' in key:
            nets.append(get_architecture(value, dm.num_classes))
            
    if len(nets) == 0:
        raise ValueError("No architecture specified")
    
    elif len(nets) == 1:
        nets = nets[0]
    
    # Si no se especifica versión, seleccionar la nueva para entrenar
    version = len(versions) if args['version'] is None else args['version']

    return args, name, exp_dir, ckpt, version, dm, nets


def get_data_module(dataset, batch_size):
    from datasets import CIFAR100DataModule, CIFAR10DataModule, ImagenetDataModule
    dataset_classes = {
        'cifar100': CIFAR100DataModule,
        'cifar10': CIFAR10DataModule,
        'imagenet': ImagenetDataModule
    }
    try:
        return dataset_classes[dataset](data_dir=f"./data/{dataset}/", batch_size=batch_size)
    except KeyError:
        raise ValueError(f"Invalid dataset: {dataset}")


def get_architecture(architecture, num_classes):
    from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
    from MobileNetV2 import mobilenet_v2
    architectures = {
        'resnet18': ResNet18,
        'resnet34': ResNet34,
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'resnet152': ResNet152,
        'mobilenetv2': mobilenet_v2
    }
    try:
        return architectures[architecture](num_classes=num_classes)
    except KeyError:
        raise ValueError(f"Invalid architecture: {architecture}")
    
def get_experiment(log_dir, architecture, dataset, version=None):
    if isinstance(architecture, list): # Si es una lista, unir los elementos
        architecture = "_".join(architecture)
    experiment_name = f"{architecture}_{dataset}"
    experiment_dir = os.path.join(log_dir, experiment_name)
    experiment_version_dir = None
    if version is not None:
        experiment_version_dir = os.path.join(experiment_dir, f"version_{version}", "checkpoints")
        if not os.path.exists(experiment_version_dir): # Verificar si el modelo existe
            raise ValueError(f"Version {version} does not exist")
        else:
            experiment_version_dir = os.path.join(experiment_version_dir, os.listdir(experiment_version_dir)[0])
    return experiment_name, experiment_dir, experiment_version_dir

def get_versions(log_dir, architecture, dataset):
    if isinstance(architecture, list): # Si es una lista, unir los elementos
        architecture = "_".join(architecture)
    experiment_dir = os.path.join(log_dir, f"{architecture}_{dataset}")
    if os.path.exists(experiment_dir):
        return os.listdir(experiment_dir)
    else:
        return []
