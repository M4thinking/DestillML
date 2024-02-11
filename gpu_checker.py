import torch

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Se encontraron {device_count} GPU(s) disponibles.")
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device_memory = torch.cuda.get_device_properties(i).total_memory
        device_memory_used = torch.cuda.memory_allocated(i)
        device_memory_free = torch.cuda.memory_reserved(i) - device_memory_used
        print("-" * 50)
        print(f"GPU {i}: {device_name}")
        print(f"Espacio de memoria total: {device_memory / 1024**3:.2f} GB")
        print(f"Memoria utilizada: {device_memory_used / 1024**3:.2f} GB")
        print(f"Memoria disponible: {device_memory_free / 1024**3:.2f} GB")
    print("-" * 50)
else:
    print("No se encontraron GPUs disponibles.")