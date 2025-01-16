import torch
from models.compare.resnet import generate_model as generate_resnet
from models.model_design_for_3d import vit_3d_resnet_fusion
from models.model_design_for_roi import *
from models.model_design_for_slice import *
def create_model(model_name):
    # 3D 数据
    if model_name == '3D_ResNet18':
        model = generate_resnet(18, n_input_channels=3, n_classes=2)
    elif model_name == '3D_ViT_ResNet18':
        model = vit_3d_resnet_fusion(18, n_input_channels=3, n_classes=2)

    # ROI 模型
    elif model_name == "ROI_transformer":
        model = TransformerClassifier(input_dim=18, d_model=64, num_heads=8, num_layers=4, dim_feedforward=256, dropout=0.1, max_len=90)

    # Slice 数据
    elif model_name == "Slice_design1":
        model = TriViewResNet18(num_classes=2)

    else:
        raise ValueError(f'Unsupported model: {model_name}')
    return model


if __name__ == '__main__':
    model = create_model('3D_ViT_ResNet18')
    x = torch.rand(1, 3, 90, 90, 90)
    out = model(x)
    print(out)