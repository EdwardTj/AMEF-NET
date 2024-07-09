import torch
import torch.nn as nn

class ViT3D(nn.Module):
    def __init__(self):
        super(ViT3D, self).__init__()
        num_classes = 2
        input_shape = (1, 182, 218, 182)
        patch_size = (14, 14, 14)
        num_layers = 8
        d_model = 1024
        num_heads = 16
        mlp_dim = 1024

        self.patch_size = patch_size
        self.embedding_dim = d_model
        self.num_patches = (input_shape[1] // patch_size[0]) * (input_shape[2] // patch_size[1]) * (input_shape[3] // patch_size[2])

        self.patch_embedding = nn.Conv3d(input_shape[0], d_model, kernel_size=patch_size, stride=patch_size)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, mlp_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        patches = self.patch_embedding(x)
        b, c, d, h, w = patches.shape
        patches = patches.view(b, c, d * h * w).permute(2, 0, 1)
        embeddings = self.transformer_encoder(patches)
        embeddings = torch.mean(embeddings, dim=0)
        output = self.fc(embeddings)
        return output

def main():
    model = ViT3D()
    # print(model)

    tmp = torch.randn(2, 1, 182, 218, 182)
    out = model(tmp)
    print('output shape:', out.shape)

    num_params = sum(p.numel() for p in model.parameters())
    print('Total number of parameters:', num_params)

    # Uncomment to calculate FLOPs using THOP library
    # from thop import profile, clever_format
    # input_size = (2, 1, 182, 218, 182)
    # flops, params = profile(model, inputs=(torch.randn(*input_size),))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(f"FLOPs: {flops}, Parameters: {params}")

if __name__ == '__main__':
    main()
