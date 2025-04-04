import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 定义源目录和目标目录
src_root = r"E:\PD_data\ppmi"
dst_root = r"D:\Code\work2-dti\data\ppmi_npz"

def process_subject(time_point, subject_id):
    """处理单个被试的数据"""
    src_path = os.path.join(src_root, time_point, "DTI_Results_GOOD", subject_id, "standard_space")
    if not os.path.exists(src_path):
        return
        
    # 创建目标目录结构
    dst_path = os.path.join(dst_root, time_point, "DTI_Results_GOOD", subject_id, "standard_space")
    os.makedirs(dst_path, exist_ok=True)
    
    # 复制所有以1mm.nii.gz结尾的文件
    for file in os.listdir(src_path):
        if file.endswith("1mm.nii.gz"):
            src_file = os.path.join(src_path, file)
            dst_file = os.path.join(dst_path, file)
            
            # 如果目标文件不存在，则复制
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)

def main():
    # 获取所有时间点目录
    time_points = ["0m", "12m", "24m"]
    
    # 获取所有被试ID
    all_subjects = set()
    for time_point in time_points:
        time_point_path = os.path.join(src_root, time_point, "DTI_Results_GOOD")
        if os.path.exists(time_point_path):
            all_subjects.update(os.listdir(time_point_path))
    
    # 创建任务列表
    tasks = [(time_point, subject) 
             for time_point in time_points 
             for subject in all_subjects]
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(lambda x: process_subject(*x), tasks), 
                 total=len(tasks), 
                 desc="Processing subjects"))

if __name__ == "__main__":
    main()