import torch
from models.compare.slowfast import SlowFast
from models.compare.C3D import C3D
from models.compare.densnet import DenseNet
from models.compare.resnet import generate_model as generate_resnet
from models.model_design import resnet_with_mil, ROIVisionTransformer, dual_resnet_transformer
def create_model(model_name):
    if model_name == 'ResNet18':
        model = generate_resnet(18, n_input_channels=3, n_classes=2)
    elif model_name == 'ResNet50':
        model = generate_resnet(50, n_input_channels=3, n_classes=2)
    # elif model_name == 'SlowFast':
    #     model = SlowFast(layers=[3, 4, 6, 3], class_num=2, dropout=0.5)
    # elif model_name == 'C3D':
    #     model = C3D(num_classes=2)
    elif model_name == 'DenseNet264':#一共有四种[121, 169, 201, 264]
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 64, 48),
                         n_input_channels=3, num_classes=2)
    elif model_name == 'DenseNet121':#一共有四种[121, 169, 201, 264]
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         n_input_channels=3, num_classes=2)
    elif model_name == "DTI_design1":   # 这个是 LPD + CSF 两个模块融合的结果 考虑下一个design是做LPD 直接output 结合 critic来决定损失权重
        model = resnet_with_mil(18, n_input_channels=3, n_classes=2)
    elif model_name == "DTI_design2":
        model = ROIVisionTransformer(roi_feature_dim=1024, voxel_feature_dim=256, num_heads=1, num_layers=1)
    elif model_name == "Dual_1":
        model = dual_resnet_transformer(18, n_input_channels=3,n_classes=2)
    else:
        raise ValueError(f'Unsupported model: {model_name}')
    return model


if __name__ == '__main__':
    model = create_model('DTI_design2')
    x = torch.rand(1, 3, 180, 180, 180)
    # label = torch.randint(0, 1, (4,))
    out = model(x)
    print(out)