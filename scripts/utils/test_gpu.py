import torch
import time
torch.cuda.set_device(4)
a = torch.randn(1000, 1000, device='cuda')
time.sleep(10)
