import torch
from models.compare.resnet import generate_model as generate_resnet
from models.model_design_for_3d import vit_3d_resnet_fusion, vit_3d_resnet_fusion_contrastive_learning
from models.contrastive_model1 import FA_MD_contrastive_model
from models.graph_model import GCNClassifier, GATClassifier, GraphSAGEClassifier
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

    # 图结构数据
    elif model_name == 'graph_model_gcn':
        model = GCNClassifier(in_channels=18, hidden_channels=64, out_channels=2)
    elif model_name == 'graph_model_gat':
        model = GATClassifier(in_channels=18, hidden_channels=64, out_channels=2)
    elif model_name == 'graph_model_graphsage':
        model = GraphSAGEClassifier(in_channels=18, hidden_channels=64, out_channels=2)
    else:
        raise ValueError(f'Unsupported model: {model_name}')
    return model


if __name__ == '__main__':
    # model = create_model('contrastive_model1')
    # x = torch.rand(2, 1, 180, 180, 180)
    # logit,map,emb = model(x)
    # print(emb)

    model = create_model('graph_model_gat')
    from torch_geometric.data import Data

    x = torch.rand((90, 18))  # 5个节点，每个节点2维特征
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long)  # 图的边列表
    edge_weight = torch.ones(edge_index.size(1))  # 权重全为1
    y = torch.tensor([1, 0, 1, 0, 1], dtype=torch.long)  # 标签
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
    output = model(data)
    print(output)
