import numpy as np
import matplotlib.pyplot as plt

# 加载热激活图数据
cam_data = np.load("./results/NCvsPD/20250103-125411/cam_fold1_sample0_image2.npy")  # 热激活图数据 (depth, height, width)
print(cam_data.shape) # 91 109 91
# 手动选择切面
slice_idx = 45  # 选择第 10 个切片
plt.imshow(cam_data[slice_idx, ...], cmap='jet')  # 显示热激活图的一个切片
plt.colorbar()
plt.title(f"Heatmap (Slice {slice_idx})")
plt.axis('off')
plt.show()