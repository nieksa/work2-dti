import os
import numpy as np
import nibabel as nib
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed


def center_crop(image, target_size):
    """
    对输入的 3D 图像进行中心裁剪。
    :param image: 输入的 3D 图像（numpy 数组）
    :param target_size: 目标裁剪尺寸，例如 (180, 180, 180)
    :return: 裁剪后的图像
    """
    depth, height, width = image.shape
    start_x = (width - target_size[0]) // 2
    start_y = (height - target_size[1]) // 2
    start_z = (depth - target_size[2]) // 2
    cropped_image = image[
                    start_z:start_z + target_size[0],
                    start_y:start_y + target_size[1],
                    start_x:start_x + target_size[2]
                    ]
    return cropped_image


def process_subject(subject_id, timepoint, input_root, output_root, target_size):
    """
    处理单个 subject 的数据
    :param subject_id: subject ID
    :param timepoint: 时间点
    :param input_root: 输入根目录
    :param output_root: 输出根目录
    :param target_size: 目标裁剪尺寸
    """
    # 构建输入路径
    input_path = os.path.join(
        input_root,
        timepoint,
        "DTI_Results_GOOD",
        subject_id,
        "standard_space"
    )

    # 查找 FA、L1、MD 文件
    fa_file = os.path.join(input_path, f"{subject_id}_FA_*_2mm.nii.gz")
    l1_file = os.path.join(input_path, f"{subject_id}_L1_*_2mm.nii.gz")
    md_file = os.path.join(input_path, f"{subject_id}_MD_*_2mm.nii.gz")

    # 使用 glob 查找匹配的文件
    fa_files = glob.glob(fa_file)
    l1_files = glob.glob(l1_file)
    md_files = glob.glob(md_file)

    # 检查是否找到文件
    if not fa_files or not l1_files or not md_files:
        print(f"警告：未找到 {subject_id} 在 {timepoint} 的完整文件，跳过该 subject。")
        return

    # 加载 FA、L1、MD 文件
    fa_data = nib.load(fa_files[0]).get_fdata()
    l1_data = nib.load(l1_files[0]).get_fdata()
    md_data = nib.load(md_files[0]).get_fdata()

    # 中心裁剪
    fa_cropped = center_crop(fa_data, target_size)
    l1_cropped = center_crop(l1_data, target_size)
    md_cropped = center_crop(md_data, target_size)

    # 合并为一个 4D 数组，并转换为 float32
    merged_data = np.stack([fa_cropped, l1_cropped, md_cropped], axis=0).astype(np.float32)
    print(f"合并后的数据形状: {merged_data.shape}")
    print(f"合并后的数据类型: {merged_data.dtype}")

    # 构建输出路径
    output_path = os.path.join(
        output_root,
        timepoint,
        "DTI_Results_GOOD",
        subject_id,
        "standard_space"
    )
    os.makedirs(output_path, exist_ok=True)

    # 保存为 npz 文件
    output_file = os.path.join(output_path, f"{subject_id}_FA_L1_MD_2mm.npz")
    np.savez_compressed(output_file, data=merged_data)
    print(f"保存合并后的数据到: {output_file}")


def process_and_save_nii_files(input_root, output_root, timepoints, target_size=(90, 90, 90)):
    """
    根据指定路径找到 3 个 nii.gz 文件，执行中心裁剪和合并操作，保存为 npz 文件。
    :param input_root: 输入根目录（例如 ./ppmi）
    :param output_root: 输出根目录（例如 E:/ppmi）
    :param timepoints: 时间点列表（例如 ["0m", "12m", "14m"]）
    :param target_size: 目标裁剪尺寸
    """
    for timepoint in timepoints:
        # 构建 DTI_Results_GOOD 路径
        dti_results_path = os.path.join(input_root, timepoint, "DTI_Results_GOOD")

        # 获取所有 subject_id 子目录
        subject_dirs = [d for d in os.listdir(dti_results_path) if os.path.isdir(os.path.join(dti_results_path, d))]

        # 使用线程池处理每个 subject
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_subject, subject_id, timepoint, input_root, output_root, target_size)
                for subject_id in subject_dirs
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"处理失败: {e}")


# 示例调用
input_root = "./ppmi"
output_root = "E://ppmi_npz2"
timepoints = ["0m", "12m", "24m"]  # 时间点
process_and_save_nii_files(input_root, output_root, timepoints)