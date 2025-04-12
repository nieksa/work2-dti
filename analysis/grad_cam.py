import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from models.utils import create_model




class FAWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model_fa  # FA 分支
        self.classifier = model.fa_classifier  # 假设最终是统一分类头

    def forward(self, x):
        embedding, _ = self.encoder(x)
        logit = self.classifier(embedding)
        return logit

class MRIWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model_mri  # GM 分支
        self.classifier = model.mri_classifier

    def forward(self, x):
        embedding, _ = self.encoder(x)
        logit = self.classifier(embedding)
        return logit


class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook (backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()

        input_tensor.requires_grad = True
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        # 权重: GAP over gradients
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)  # (B,C,1,1,1)
        cam = torch.sum(weights * self.activations, dim=1)  # (B,H,W,D)

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.unsqueeze(0)
        cam_resized = F.interpolate(cam, size=input_tensor.shape[2:], mode='trilinear', align_corners=False)
        return cam_resized


model_path = "./analysis/model/contrastive_model2_fold_3_epoch_101_20250407_161208_acc_0.8980.pth"
model = create_model("contrastive_model2")
state_dict = torch.load(model_path, map_location='cpu')
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '') if k.startswith('module.') else k
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)
model.eval()


fa_path = "./analysis/data/003102_FA_4normalize_to_target_1mm.nii.gz"
fa_nii = nib.load(fa_path)
fa_affine = fa_nii.affine
fa = fa_nii.get_fdata()
fa = fa[None, ...]
fa_tensor = torch.from_numpy(fa).float()
fa_tensor = fa_tensor.unsqueeze(0)

target_class = 1

fa_model = FAWrapper(model)
fa_target_layer = fa_model.encoder.layer1[-1]
fa_grad_cam = GradCAM3D(fa_model, fa_target_layer)
fa_heatmap = fa_grad_cam(fa_tensor, class_idx=target_class)


fa_nifti = nib.Nifti1Image(fa_heatmap.squeeze().cpu().numpy(), affine=fa_affine)
nib.save(fa_nifti, f"./analysis/data/fa_gradcam_heatmap_target_{target_class}.nii.gz")

# mri_nifti = nib.Nifti1Image(mri_heatmap.squeeze().cpu().numpy(), affine=fa_affine)
# nib.save(mri_nifti, f"./analysis/data/mri_gradcam_heatmap_target_{target_class}.nii.gz")
