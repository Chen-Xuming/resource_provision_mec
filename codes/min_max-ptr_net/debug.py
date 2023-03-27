import torch

# 创建一个 5x2 的张量
tensor = torch.tensor([1, 2, 3, 4, 5])

print(tensor)

# 将张量复制 100 次
tensor = tensor.repeat(10, 1)

print(tensor)
print(tensor.size(1))

# # 将张量沿着第 0 维堆叠在一起
# new_tensor = torch.cat([tensor], dim=0)
#
# # 输出新张量的形状
# print(new_tensor.shape)  # 输出 torch.Size([500, 2])
