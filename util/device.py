import torch

def get_device(device_id=None, min_free_gb=10):
    if not torch.cuda.is_available():
        raise RuntimeError(f"No GPU found")

    if device_id is None or device_id == -1:
    
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu = torch.device(f'cuda:{i}')
            free_memory, total_memory = torch.cuda.mem_get_info(i)
            free_memory_gb = free_memory / 1024**3

            if free_memory_gb >= min_free_gb:
                props = torch.cuda.get_device_properties(i)
                print(f"Using GPU {i}: {props.name} with {free_memory_gb:.2f}GB free")
                return gpu
        
        raise RuntimeError(f"No GPU found with at least {min_free_gb} GB of free VRAM.")
    
    props = torch.cuda.get_device_properties(device_id)
    gpu = torch.device(f'cuda:{device_id}')
    free_memory, total_memory = torch.cuda.mem_get_info(gpu)
    total_memory = int(total_memory / 1024**3)
    free_memory = int(free_memory / 1024**3)  
    print(f"Using GPU {device_id}: {props.name} with {free_memory:.2f}GB")
    return gpu