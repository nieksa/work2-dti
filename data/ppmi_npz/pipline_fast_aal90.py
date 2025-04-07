# -*- coding: utf-8 -*-
import os
import re
import glob
import subprocess
import nibabel as nib
import numpy as np
from multiprocessing import Pool, cpu_count

AAL_TEMPLATE = "./AAL90_2mm.nii"
NUM_REGIONS = 90
FSL_FAST_PATH = "fast"  # 确保已加入环境变量

def find_target_files():
    return glob.glob("./[012]*m/DTI_Results_GOOD/*/T1/co_*_swap_bet_crop_resample_2MNI152.nii.gz")

def extract_subject_id(filename):
    m = re.match(r'^co_(.+)_swap_bet_crop_resample_2MNI152\.nii\.gz$', os.path.basename(filename))
    return m.group(1) if m else None

def run_fast(input_file, output_prefix):
    cmd = [FSL_FAST_PATH, "-t", "1", "-n", "3", "-o", output_prefix, input_file]
    print("Running FAST:", " ".join(cmd))
    subprocess.call(cmd)

def move_output(tmp_prefix, seg_dir, subject_id):
    out_gm = os.path.join(seg_dir, "co_%s_swap_bet_crop_resample_2MNI152_GM.nii.gz" % subject_id)
    out_wm = os.path.join(seg_dir, "co_%s_swap_bet_crop_resample_2MNI152_WM.nii.gz" % subject_id)
    out_csf = os.path.join(seg_dir, "co_%s_swap_bet_crop_resample_2MNI152_CSF.nii.gz" % subject_id)

    os.rename(tmp_prefix + "_pve_1.nii.gz", out_gm)
    os.rename(tmp_prefix + "_pve_2.nii.gz", out_wm)
    os.rename(tmp_prefix + "_pve_0.nii.gz", out_csf)
    return out_gm, out_wm, out_csf

def extract_by_template(seg_path, template_path, out_txt):
    seg_img = nib.load(seg_path)
    template_img = nib.load(template_path)

    seg_data = seg_img.get_data()
    template_data = template_img.get_data()

    if seg_data.shape != template_data.shape:
        raise ValueError("Shape mismatch between seg and template")

    region_vals = []
    for i in range(1, NUM_REGIONS + 1):
        mask = (template_data == i)
        if np.any(mask):
            region_vals.append(np.mean(seg_data[mask]))
        else:
            region_vals.append(0.0)

    np.savetxt(out_txt, np.array(region_vals).reshape(-1, 1), fmt="%.6f")

def clean_temp_files(prefix):
    for suffix in ["_pve_0.nii.gz", "_pve_1.nii.gz", "_pve_2.nii.gz", "_pveseg.nii.gz", "_mixeltype.nii.gz", "_seg.nii.gz"]:
        tmp_file = prefix + suffix
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

def process_subject(target_file):
    subject_id = extract_subject_id(target_file)
    if not subject_id:
        return "Invalid filename: %s" % target_file  # 使用 % 格式化

    subject_dir = os.path.dirname(target_file)
    base_dir = os.path.dirname(subject_dir)
    seg_dir = os.path.join(base_dir, "T1_seg")
    txt_dir = os.path.join(base_dir, "T1_txt")
    if not os.path.exists(seg_dir): os.makedirs(seg_dir)
    if not os.path.exists(txt_dir): os.makedirs(txt_dir)

    tmp_prefix = "tmp_%s" % subject_id
    run_fast(target_file, tmp_prefix)

    try:
        gm_path, wm_path, csf_path = move_output(tmp_prefix, seg_dir, subject_id)

        extract_by_template(gm_path, AAL_TEMPLATE, os.path.join(txt_dir, "%s_GM.txt" % subject_id))
        extract_by_template(wm_path, AAL_TEMPLATE, os.path.join(txt_dir, "%s_WM.txt" % subject_id))
        extract_by_template(csf_path, AAL_TEMPLATE, os.path.join(txt_dir, "%s_CSF.txt" % subject_id))

        clean_temp_files(tmp_prefix)
        return "✓ Completed subject: %s" % subject_id  # 使用 % 格式化
    except Exception as e:
        return "✗ Failed subject: %s, Error: %s" % (subject_id, str(e))  # 使用 % 格式化


def main():
    target_files = find_target_files()
    if not os.path.exists(AAL_TEMPLATE):
        print("AAL90 template not found at:", AAL_TEMPLATE)
        return

    # 使用 Pool 来并行处理任务
    pool = Pool(processes=cpu_count())  # 使用 CPU 核数自动匹配
    results = pool.map(process_subject, target_files)
    
    # 打印处理结果
    for result in results:
        print(result)

if __name__ == '__main__':
    main()
