import torch
from models.compare.slowfast import SlowFast
from models.compare.C3D import C3D
from models.compare.densnet import DenseNet
from models.compare.resnet import generate_model as generate_resnet
from models.model_design import (generate_resnet_vit, generate_resnet_cotatt, generate_resnet_simam,
                                 generate_resnet_tripletattention, generate_MDL)
from models.model_design import resnet_with_mil
def create_model(model_name):
    if model_name == 'ResNet18':
        model = generate_resnet(18, n_input_channels=1, n_classes=2)
    elif model_name == 'ResNet50':
        model = generate_resnet(50, n_input_channels=1, n_classes=2)
    elif model_name == 'SlowFast':
        model = SlowFast(layers=[3, 4, 6, 3], class_num=2, dropout=0.5)
    elif model_name == 'C3D':
        model = C3D(num_classes=2)
    elif model_name == 'DenseNet264':#一共有四种[121, 169, 201, 264]
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 64, 48),
                         n_input_channels=1, num_classes=2)
    elif model_name == 'DenseNet121':#一共有四种[121, 169, 201, 264]
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         n_input_channels=1, num_classes=2)
    elif model_name == 'Design2':   # CCT 的思路，通过三个resnet的3D卷积块降低尺度256, 8, 8, 8，然后ViT,patch1*1*1作为一个voxel
        model = generate_resnet_vit(18)
    elif model_name == 'Design4':   # layer + cot attention
        model = generate_resnet_cotatt(18)
    elif model_name == 'Design5':   # layer + simama轻量级注意力机制
        model = generate_resnet_simam(18)
    elif model_name == 'Design6':   # layer + triplet attention 多轴向注意力
        model = generate_resnet_tripletattention(18)
    elif model_name == "Design10":  # MDL可以尝试调一下
        model = generate_MDL(model_depth=18,in_planes=1,num_classes=2)
    elif model_name == "DTI_design1":   # 这个是 LPD + CSF 两个模块融合的结果 考虑下一个design是做LPD 直接output 结合 critic来决定损失权重
        model = resnet_with_mil(18)
    else:
        raise ValueError(f'Unsupported model: {model_name}')
    return model


if __name__ == '__main__':
    model = create_model('DTI_design1')
    x = torch.rand(4, 1, 91, 109, 91)
    label = torch.randint(0, 2, (4,))
    out = model(x)
    print(out)