from nilearn import plotting
import matplotlib.pyplot as plt
import nibabel as nib
# 可视化路径
heatmap_nii_path = "./analysis/data/fa_gradcam_heatmap_target_1.nii.gz"
# heatmap_nii_path = "./analysis/mri_gradcam_heatmap_target_1.nii.gz"
mni_1mm_path = "./analysis/template/MNI152_T1_1mm_brain.nii"
mni_2mm_path = "./analysis/template/MNI152_T1_2mm_brain.nii"
# 加载 Nifti 图像
display_img = nib.load(heatmap_nii_path)
mni_1mm_img = nib.load(mni_1mm_path)
mni_2mm_img = nib.load(mni_2mm_path)

# ===============================
# 1. 玻璃脑图（Glass Brain）
# ===============================
plotting.plot_glass_brain(
    display_img,
    display_mode='lyrz',
    colorbar=True,
    cmap=plt.cm.plasma,
    threshold=0.7,  # 可根据实际情况调整
    title='FA Grad-CAM Glass Brain'
)
plt.show()

# ===============================
# 2. 三视图切片（横断，冠状，矢状）
# ===============================
# plotting.plot_stat_map(
#     display_img,
#     mni_1mm_img,
#     display_mode='ortho',  # 分别是 x（矢状），y（冠状），z（横断）
#     threshold=0.7,
#     colorbar=True,
#     cmap='hot',
#     title='FA Grad-CAM Ortho Slices'
# )
# plt.show()

# ===============================
# 3. 大脑皮层
# ===============================
# html_view = plotting.view_img_on_surf(
#     display_img,
#     surf_mesh='fsaverage',
#     threshold=0.7,
#     title='FA Grad-CAM Surface',
#     cmap='hot'  # 也可以是 'coolwarm', 'jet', 'plasma', 'viridis' 等
# )
# html_view.open_in_browser()

# ===============================
# 4. ROI
# ===============================
import numpy as np
aal_116_path = "./analysis/atlases/aal116MNI.nii.gz"
aal_116_index_path = "./analysis/atlases/aal116NodeIndex.1D"
aal_116_name_path = "./analysis/atlases/aal116NodeNames.txt"
aal_116_img = nib.load(aal_116_path)
aal_116_index = np.loadtxt(aal_116_index_path)
aal_116_name = np.loadtxt(aal_116_name_path, dtype=str)

from nilearn.image import resample_to_img
# display_img = resample_to_img(display_img, aal_116_img, interpolation='nearest', force_resample=True, copy_header=True)
display_img = resample_to_img(display_img, aal_116_img, interpolation='nearest', force_resample=True)
heatmap_data = display_img.get_fdata()
aal_data = aal_116_img.get_fdata()


results = []
label_to_name = {
    int(index): name
    for index, name in zip(aal_116_index, aal_116_name)
}

activation_map = np.zeros(aal_116_img.shape)

for label in label_to_name:
    mask = aal_data == label
    region_values = heatmap_data[mask]

    if region_values.size > 0:
        region_sum = np.sum(region_values)
        region_mean = np.mean(region_values)
        results.append((label, region_sum, region_mean))

        activation_map[mask] = region_mean
# results_sorted = sorted(results, key=lambda x: x[1], reverse=True)


activation_map_img = nib.Nifti1Image(activation_map, aal_116_img.affine)
html_view = plotting.view_img_on_surf(
    activation_map_img,
    surf_mesh='fsaverage',
    title='ROI',
    cmap='inferno', 
)
html_view.open_in_browser()


