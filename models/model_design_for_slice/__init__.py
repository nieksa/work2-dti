# 这里设计原则是模型输入是一个 三元组
# forward(self, x):
#     x_slice = x[0]
#     y_slice = x[1]
#     z_slice = x[2]
#     每个slice都是(3,180,180)
from .slice_design1 import TriViewResNet18