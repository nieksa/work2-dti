import torch
from models.compare.resnet import generate_model as generate_resnet
from models.model_design_for_3d import vit_3d_resnet_fusion, vit_3d_resnet_fusion_contrastive_learning
from models.contrastive_model1 import FA_MD_contrastive_model
def create_model(model_name):
    # 3D 数据
    if model_name == '3D_ResNet18':
        model = generate_resnet(18, n_input_channels=3, n_classes=2)
    elif model_name == '3D_ViT_ResNet18':
        model = vit_3d_resnet_fusion(18, n_input_channels=3, n_classes=2)
    # 对比学习模型单通道融合机制
    elif model_name == '3D_ViT_ResNet18_Contrastive_Learning':
        model = vit_3d_resnet_fusion_contrastive_learning(18, n_input_channels=1, n_classes=2)
    elif model_name == 'contrastive_model1':
        model = FA_MD_contrastive_model(18,n_input_channels=1, n_classes=2)

    else:
        raise ValueError(f'Unsupported model: {model_name}')
    return model


if __name__ == '__main__':
    model = create_model('contrastive_model1')
    x = torch.rand(2, 1, 180, 180, 180)
    logit,map,emb = model(x)
    print(emb)