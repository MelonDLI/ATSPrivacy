import Models
import utils
import torchvision
import torch
from preprocess import preprocess
from optimization_strategy import training_strategy

setup = utils.system_startup()

# Data
defs = training_strategy('conservative'); defs.epochs = 200 #conservative
loss_fn, trainloader, validloader = preprocess(defs=defs)
data, target = next(iter(trainloader)) # single batch
print(data.shape)
data = data.to(**setup)
# target = target.to(**setup)

# Model
filename = '/home/remote/u7076589/ATSPrivacy/Melon/checkpoints/MoEx_bn_ResNet_20_lam_09_p_05/ResNet20-4_200.pth'

model = Models.ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=100, num_channels=3, base_width=16 * 4)
model.to(**setup)

model.load_state_dict(torch.load(filename))
# print(model)

# data feed into model
model.eval()
output = model.conv1(data) #torch.Size([128, 64, 32, 32])
output = model.bn1(output)  #torch.Size([128, 64, 32, 32])
print(output.shape)
output = model.relu(output)
print(output.shape)
print('layers:')

for layer in model.layers:
    output = layer(output)
    print(output.shape)
print('----------')
output = model.pool(output)
print(output.shape)
output = torch.flatten(output,1)
print(output.shape)
output = model.fc(output)
print(output.shape)


# torch.Size([128, 3, 32, 32])
# torch.Size([128, 64, 32, 32])
# torch.Size([128, 64, 32, 32])
# layers:
# torch.Size([128, 64, 32, 32])
# torch.Size([128, 128, 16, 16])
# torch.Size([128, 256, 8, 8])  hc last intermediate representation before the final pooling that aggregates allk the spatial information
# ----------
# torch.Size([128, 256, 1, 1]) C: global average pooling followed by one or more fully-connected layers
# torch.Size([128, 256])
# torch.Size([128, 100])

# ResNet(
#   (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (layers): ModuleList(
#     (0): Sequential(
#       (0): BasicBlock(
#         (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (1): BasicBlock(
#         (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (2): BasicBlock(
#         (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Sequential(
#       (0): BasicBlock(
#         (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (downsample): Sequential(
#           (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
#           (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#       (1): BasicBlock(
#         (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (2): BasicBlock(
#         (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (2): Sequential(
#       (0): BasicBlock(
#         (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (downsample): Sequential(
#           (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
#           (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#       (1): BasicBlock(
#         (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (2): BasicBlock(
#         (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#   )
#   (pool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (fc): Linear(in_features=256, out_features=100, bias=True)
#   (pono): PONO()
#   (ms): MomentShortcut()
# )