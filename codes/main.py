import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn

probs = torch.tensor([
    [[0.5, 0.2, 0.3]],
    [[0.1, 0.4, 0.5]]
], dtype=torch.float)
print(probs, probs.shape)
print(probs[0], probs[1])
fp = torch.mm(probs[0].reshape(3, 1), probs[1].reshape(1, 3))
print(fp, fp.shape)

fp = fp.flatten()
print(fp, fp.shape)
