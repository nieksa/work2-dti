import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def find_brain_boundaries(data, threshold=0):
    """
    找到脑区数据的边界。
    :param data: 3D 图像数据 (depth, height, width)
    :param threshold: 背景阈值，小于该值的像素被认为是背景
    :return: (depth_start, depth_end), (height_start, height_end), (width_start, width_end)
    """
    # 找到 depth 维度的边界
    depth_sum = np.sum(data, axis=(1, 2))  # 沿 height 和 width 求和
    depth_start = np.argmax(depth_sum > threshold)  # 第一个非背景切片
    depth_end = len(depth_sum) - np.argmax(depth_sum[::-1] > threshold)  # 最后一个非背景切片

    # 找到 height 维度的边界
    height_sum = np.sum(data, axis=(0, 2))  # 沿 depth 和 width 求和
    height_start = np.argmax(height_sum > threshold)
    height_end = len(height_sum) - np.argmax(height_sum[::-1] > threshold)

    # 找到 width 维度的边界
    width_sum = np.sum(data, axis=(0, 1))  # 沿 depth 和 height 求和
    width_start = np.argmax(width_sum > threshold)
    width_end = len(width_sum) - np.argmax(width_sum[::-1] > threshold)

    return (depth_start, depth_end), (height_start, height_end), (width_start, width_end)


def extract_brain_region(data, boundaries):
    """
    提取包含脑区数据的最大长方体。
    :param data: 3D 图像数据 (depth, height, width)
    :param boundaries: 脑区边界，格式为 ((depth_start, depth_end), (height_start, height_end), (width_start, width_end))
    :return: 提取后的 3D 图像数据
    """
    (depth_start, depth_end), (height_start, height_end), (width_start, width_end) = boundaries
    return data[depth_start:depth_end, height_start:height_end, width_start:width_end]

def plot_slices(data, title="Slices"):
    """
    绘制 3D 数据的中间切片。
    :param data: 3D 图像数据
    :param title: 图像标题
    """
    depth, height, width = data.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 绘制 depth 维度的中间切片
    axes[0].imshow(data[depth // 2, :, :], cmap="gray")
    axes[0].set_title(f"Depth Slice {depth // 2}")

    # 绘制 height 维度的中间切片
    axes[1].imshow(data[:, height // 2, :], cmap="gray")
    axes[1].set_title(f"Height Slice {height // 2}")

    # 绘制 width 维度的中间切片
    axes[2].imshow(data[:, :, width // 2], cmap="gray")
    axes[2].set_title(f"Width Slice {width // 2}")

    plt.suptitle(title)
    plt.show()

def main():
    # 加载 .nii.gz 文件
    # "./data/ppmi/0m/DTI_Results_GOOD/003101/standard_space/"
    file_path = "ppmi/0m/DTI_Results_GOOD/003101/standard_space/003101_FA_4normalize_to_target_1mm.nii.gz"
    nii_image = nib.load(file_path)

    # 获取数据数组
    data = nii_image.get_fdata()
    print("Data shape:", data.shape)  # 输出数据的形状，例如 (91, 109, 91)

    # 找到脑区边界
    boundaries = find_brain_boundaries(data)

    # 提取脑区数据
    brain_region = extract_brain_region(data, boundaries)
    print("Brain region shape:", brain_region.shape)  # 输出提取后的形状

    plot_slices(data, title="Original Data Slices")
    plot_slices(brain_region, title="Brain Region Slices")


if __name__ == "__main__":
    main()