import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn

if torch.cuda.is_available():
    print('Using GPU, %i devices.' % torch.cuda.device_count())
print(torch.backends.cudnn.enabled)
print(torch.cuda.get_device_name(0))

arr = [1, 2, 3]
arr2 = arr

print(arr2)

arr2.append(4)

print(arr)
print(arr2)