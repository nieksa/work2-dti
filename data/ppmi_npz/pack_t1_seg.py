import os
import shutil
import multiprocessing
from tqdm import tqdm
import glob

def copy_directory(src_path, dst_path):
    """
    复制目录及其内容到目标路径
    
    Args:
        src_path: 源目录路径
        dst_path: 目标目录路径
    """
    try:
        # 确保目标目录存在
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # 如果源目录存在，则复制
        if os.path.exists(src_path):
            # 如果目标目录已存在，先删除
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            
            # 复制目录及其内容
            shutil.copytree(src_path, dst_path)
            return True
        else:
            print(f"源目录不存在: {src_path}")
            return False
    except Exception as e:
        print(f"复制目录时出错 {src_path} -> {dst_path}: {str(e)}")
        return False

def process_subject(args):
    """
    处理单个受试者的目录复制
    
    Args:
        args: 包含源目录和目标目录的元组
    """
    subject_id, time_point, src_base, dst_base = args
    
    # 构建源路径和目标路径
    src_seg_path = os.path.join(src_base, time_point, "DTI_Results_GOOD", subject_id, "T1_seg")
    src_txt_path = os.path.join(src_base, time_point, "DTI_Results_GOOD", subject_id, "T1_txt")
    
    dst_seg_path = os.path.join(dst_base, time_point, "DTI_Results_GOOD", subject_id, "T1_seg")
    dst_txt_path = os.path.join(dst_base, time_point, "DTI_Results_GOOD", subject_id, "T1_txt")
    
    # 复制目录
    seg_result = copy_directory(src_seg_path, dst_seg_path)
    txt_result = copy_directory(src_txt_path, dst_txt_path)
    
    return (subject_id, time_point, seg_result, txt_result)

def main():
    # 源目录和目标目录
    src_base = "D:\Code\work2-dti\data\ppmi_npz"
    dst_base = "E:/ppmi_t1/"
    
    # 时间点
    time_points = ["0m", "12m", "24m"]
    
    # 准备任务列表 - 只包含实际存在的路径组合
    tasks = []
    
    # 遍历每个时间点目录
    for time_point in time_points:
        dti_path = os.path.join(src_base, time_point, "DTI_Results_GOOD")
        if os.path.exists(dti_path):
            # 遍历该时间点下的所有受试者目录
            for subject_dir in os.listdir(dti_path):
                subject_path = os.path.join(dti_path, subject_dir)
                if os.path.isdir(subject_path):
                    # 检查T1_seg或T1_txt目录是否存在
                    seg_path = os.path.join(subject_path, "T1_seg")
                    txt_path = os.path.join(subject_path, "T1_txt")
                    
                    if os.path.exists(seg_path) or os.path.exists(txt_path):
                        tasks.append((subject_dir, time_point, src_base, dst_base))
    
    print(f"找到 {len(tasks)} 个需要复制的目录组合")
    
    # 使用多进程并行处理
    num_processes = multiprocessing.cpu_count()
    print(f"使用 {num_processes} 个进程进行并行处理...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_subject, tasks), total=len(tasks), desc="复制文件"))
    
    # 统计结果
    success_count = sum(1 for _, _, seg, txt in results if seg or txt)
    print(f"处理完成: 成功复制 {success_count}/{len(tasks)} 个目录")

if __name__ == "__main__":
    main()