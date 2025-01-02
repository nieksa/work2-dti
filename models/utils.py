import torch
# from torch.onnx.symbolic_opset9 import numel
# from monai.networks.nets.classifier import Classifier, Discriminator, Critic
from vit_pytorch.vit_3d import ViT
from vit_pytorch.cct_3d import cct_4
from vit_pytorch.vivit import ViT as ViViT
from vit_pytorch.simple_vit_3d import SimpleViT

from models.compare.slowfast import SlowFast
from models.compare.C3D import C3D
from models.compare.I3D import InceptionI3d
from models.compare.densnet import DenseNet
from models.compare.resnet import ResNet, Bottleneck, BasicBlock, get_inplanes
from models.compare.vgg import VGG

from models.model_design import *

# 有提升的design
# 4 5
def create_model(model_name):
    if model_name == 'ViT':
        model = ViT(image_size=128, image_patch_size=16, frames=128, frame_patch_size=16,
                    num_classes=2, dim=1024, depth=2, heads=4, mlp_dim=64, pool='cls',
                    channels=1, dim_head=32, dropout=0.2, emb_dropout=0.1)
    elif model_name == 'ResNet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_input_channels=1, n_classes=2)
    elif model_name == 'ResNet50':
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), n_input_channels=1, n_classes=2)
    elif model_name == 'ViViT': # 效果不是很好，不知道是不是参数的问题
        model = ViViT(
            image_size=128,  # image size
            frames=128,  # number of frames
            image_patch_size=16,  # image patch size
            frame_patch_size=2,  # frame patch size
            num_classes=2,
            dim=512,
            spatial_depth=5,  # depth of the spatial transformer
            temporal_depth=5,  # depth of the temporal transformer
            heads=5,
            mlp_dim=1024,
            channels = 1,
            variant='factorized_encoder',  # or 'factorized_self_attention'
        )
    elif model_name == 'SlowFast':
        model = SlowFast(layers=[3, 4, 6, 3], class_num=2, dropout=0.5)
    elif model_name == 'VGG':
        model = VGG(dropout=0.5, n_classes=2)
    elif model_name == 'C3D':
        model = C3D(num_classes=2)
    elif model_name == 'I3D':
        model = InceptionI3d(num_classes=2, spatial_squeeze=True,
                     final_endpoint='Logits', name='inception_i3d', in_channels=1, dropout_keep_prob=0.5)
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
    elif model_name == 'cct4': #复杂度高，4090 24G 跑不了
        model = cct_4(img_size=128, num_frames=128, num_classes=2, n_input_channels= 1)
    elif model_name == 'SimpleViT': #复杂度高，4090 24G 跑不了
        model = SimpleViT(
            image_size = 128,          # image size
            frames = 128,               # number of frames
            image_patch_size = 16,     # image patch size
            frame_patch_size = 2,      # frame patch size
            num_classes = 2,
            dim = 512,
            depth = 5,
            heads = 8,
            mlp_dim = 1024,
            channels = 1,
            dim_head = 64
        )
    elif model_name == 'Design1':   # 双重ResNet，提取大体素和小体素特征， concat后ViT，这个模型太大跑不动
        model = Design1(in_channels=1, out_channel=128, class_num=2, num_blocks=[1,1,1])
    elif model_name == 'Design2':   # CCT 的思路，通过三个resnet的3D卷积块降低尺度256, 8, 8, 8，然后ViT,patch1*1*1作为一个voxel
        model = generate_resnet_vit(18)
    elif model_name == 'Design3':  # Resnet每一个layer后面都接上一个 coord attention3d联合多个方向的空间特征和通道注意力
        model = generate_resnet_coordatt(18)
    elif model_name == 'Design4':   # layer + cot attention
        model = generate_resnet_cotatt(18)
    elif model_name == 'Design5':   # layer + simama轻量级注意力机制
        model = generate_resnet_simam(18)
    elif model_name == 'Design6':   # layer + triplet attention 多轴向注意力
        model = generate_resnet_tripletattention(18)
    elif model_name == 'Design7':   # 双分支，transformer编解码256 8 8 8 concat resnet256 8 8 8 = 512 8 8 8通道融合后fc
        model = dual_mix_1(18)
    elif model_name == 'Design8':   # 双分支一路resnet 一路transformer, concat triplet attention pooling fc
        model = generate_resnet_transformer_voxel(18)
    elif model_name == "Design10":  # MDL单模态但是分3路
        model = generate_MDL(model_depth=18,in_planes=1,num_classes=2)
    elif model_name == "Design11":  # MDL只有一个分支
        model = generate_MDL_1_branch(model_depth=18,in_planes=1,num_classes=2)
    else:
        raise ValueError(f'Unsupported model: {model_name}')
    return model


# model_name = ['Classifier','ResNet18','ResNet50','C3D','I3D','DenseNet264','DenseNet121','SlowFast','VGG',
#               'ViT','cct4','ViViT','SimpleViT']

if __name__ == '__main__':
    model = create_model('ResNet18')
    x = torch.rand(4, 1, 121, 91, 121)
    label = torch.randint(0, 2, (4,))
    out = model(x)
    print(out)