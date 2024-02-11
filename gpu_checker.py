import torch

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Se encontraron {device_count} GPU(s) disponibles.")
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device_memory = torch.cuda.get_device_properties(i).total_memory
        print("-" * 50)
        print(f"GPU {i}: {device_name}")
        print(f"Espacio de memoria: {device_memory / 1024**3:.2f} GB")
    print("-" * 50)
else:
    print("No se encontraron GPUs disponibles.")