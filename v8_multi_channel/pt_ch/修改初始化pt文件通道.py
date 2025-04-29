from ultralytics import YOLO
import torch
import torch.nn as nn

# 加载原模型
model = YOLO('yolov8n.pt')  # 或 yolov8s.pt, yolov8m.pt 等

# 找到模型 backbone 的第一层
first_layer = model.model.model[0]  # 通常是一个 Conv 层

# 保存旧权重
old_conv = first_layer.conv  # nn.Conv2d
old_weight = old_conv.weight.data  # [out_c, 3, kH, kW]

# 创建新的 Conv 层，输入通道为8，其余参数一致
new_conv = nn.Conv2d(
    in_channels=8,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=old_conv.bias is not None
)

# # 初始化新权重，保留原3通道，后5通道设为0
# with torch.no_grad():
#     new_conv.weight[:, :3, :, :] = old_weight
#     new_conv.weight[:, 3:, :, :] = 0.0
#     if old_conv.bias is not None:
#         new_conv.bias = old_conv.bias  # 拷贝偏置
# 使用原始权重扩展填满 8 通道（循环方式）
with torch.no_grad():
    for i in range(8):
        new_conv.weight[:, i, :, :] = old_weight[:, i % 3, :, :]  # 循环复制 R,G,B
    if old_conv.bias is not None:
        new_conv.bias.copy_(old_conv.bias)
# 替换第一层
first_layer.conv = new_conv

# 保存修改后的模型
torch.save(model, 'yolov8n_8.pt')

