import torch
import torch.nn as nn
import torch.nn.functional as F

class ViT3D(nn.Module):
    def __init__(self):
        super(ViT3D, self).__init__()
        num_classes = 2  # 分类类别数
        input_shape = (1, 182, 218, 182)  # 输入图像形状（通道数、高度、宽度）
        patch_size = (14, 14, 14)  # 补丁大小
        num_layers = 8  # Transformer层数
        d_model = 256  # Transformer中嵌入向量的维度
        num_heads = 16  # 自注意力头的数量
        mlp_dim = 1024
        self.patch_size = patch_size
        self.embedding_dim = d_model
        self.num_patches = (input_shape[1] // patch_size[0]) * (input_shape[2] // patch_size[1]) * (input_shape[3] // patch_size[2])
        self.patch_embedding = nn.Conv3d(input_shape[0], d_model, kernel_size=patch_size, stride=patch_size)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, num_heads, mlp_dim), num_layers)
        self.fc = nn.Linear(d_model, num_classes)



    def forward(self, x):
        patches = self.patch_embedding(x)
        embeddings = patches.flatten(2).transpose(1, 2)
        embeddings = self.transformer_encoder(embeddings)
        embeddings = embeddings.mean(1)
        output = self.fc(embeddings)
        return output
#
# def main():
#     model = ViT3D()
#     # print(model)
#     tmp = torch.randn(2, 1, 182, 182, 182)
#     out = model(tmp)
#     print('resnet:', out.shape)
#
#     p = sum(map(lambda p: p.numel(), model.parameters()))
#     print('parameters size:', p)
#
#
# if __name__ == '__main__':
#     main()

import torch
from thop import profile
from thop import clever_format
def main():
    model = ViT3D()
    print(model)
    tmp = torch.randn(2, 1, 182, 218, 182)
    out = model(tmp)
    print('resnet:', out.shape)
    # input_size = (2, 1, 182, 182, 182)
    # input_data = torch.randn(input_size)
    # flops, params = profile(model, inputs=(input_data,))
    # flops = clever_format(flops, "%.3f")
    # print(f"FLOPs: {flops}","parameter:",params)
    p = sum(map(lambda p: p.numel(), model.parameters()))
    print('parameters size:', p)

if __name__ == '__main__':
    main()