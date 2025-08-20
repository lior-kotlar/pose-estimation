import numpy as np
import torch
import json
from torchsummary import summary
import torch
from torch import nn
import math
from pytorch_vit_encoder import CustomViT
from CNNs import Decoder2d


class CNN_Decoder(nn.Module):
    def __init__(self, num_output_channels,
                 kernel_size,
                 num_base_filters,
                 projection_dim):
        super(CNN_Decoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.num_base_filters = num_base_filters
        self.projection_dim = projection_dim
        self.deconv1 = nn.ConvTranspose2d(in_channels=self.projection_dim, out_channels=self.projection_dim,
                                          kernel_size=self.kernel_size,
                                          stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=self.projection_dim, out_channels=self.projection_dim,
                                          kernel_size=self.kernel_size,
                                          stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=self.projection_dim, out_channels=self.projection_dim,
                                          kernel_size=self.kernel_size,
                                          stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=self.projection_dim, out_channels=self.num_output_channels,
                                          kernel_size=self.kernel_size,
                                          stride=2, padding=1, output_padding=1)
        self.leakyrelu = nn.LeakyReLU(0.1)
        pass

    def forward(self, x):
        x = x.view(-1, self.projection_dim, 12, 12)
        # Add deconvolution layers
        x = self.leakyrelu(self.deconv1(x))
        x = self.leakyrelu(self.deconv2(x))
        x = self.leakyrelu(self.deconv3(x))
        x = self.leakyrelu(self.deconv4(x))
        x = self.normalize_between_0_and_1(x)
        return x

    def get_conv2d_transpose(self, in_channels, out_channels, stride):
        conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=self.kernel_size,
                                  stride=stride, padding=1, output_padding=1)
        nn.init.xavier_normal_(conv.weight)
        return conv

    @staticmethod
    def normalize_between_0_and_1(x):
        x = (x - x.min()) / (x.max() - x.min())
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # self.WQ = nn.Linear(d_model, d_model)
        # self.WK = nn.Linear(d_model, d_model)
        # self.WV = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.ReLU(),
        )

    def forward(self, x):
        # query = self.WQ(x)
        # key = self.WK(x)
        # value = self.WV(x)
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        ffn_output = self.ffn(x)
        return self.norm2(ffn_output + x)


class ViTEncoder(nn.Module):
    def __init__(self,
                 image_size=192,
                 num_input_channels=4,
                 patch_size=16,
                 num_layers=8,
                 num_heads=8,
                 d_model=512):
        super(ViTEncoder, self).__init__()
        self.d_model = d_model
        self.num_input_channels = num_input_channels
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(patch_size * patch_size * num_input_channels, d_model)
        self.position_encoding = PositionalEncoding(d_model, self.num_patches)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)])

    def get_output_shape(self):
        return (self.num_patches, self.d_model)

    def forward(self, x):
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(x.shape[0],
                                self.num_input_channels * self.patch_size * self.patch_size, -1).permute(0, 2, 1)
        x = self.patch_embedding(x)
        x = self.position_encoding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_output_channels, patch_size, num_patches):
        super().__init__()
        self.d_model = d_model
        self.num_output_channels = num_output_channels
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.sqrt_num_patches = int(math.sqrt(num_patches))

        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.linear3 = nn.Linear(d_model, patch_size ** 2 * num_output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        B, _, _ = x.size()
        x1 = self.relu(self.linear1(x)) + x
        x2 = self.relu(self.linear2(x1)) + x1
        x = self.relu(self.linear3(x2))
        x = x.view(B, self.sqrt_num_patches,
                   self.sqrt_num_patches,
                   self.patch_size,
                   self.patch_size,
                   self.num_output_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.num_output_channels,
                   self.sqrt_num_patches * self.patch_size,
                   self.sqrt_num_patches * self.patch_size)
        return x


class VIT_encoder_decoder(nn.Module):
    def __init__(self, config, image_size, number_of_output_channels):
        super(VIT_encoder_decoder, self).__init__()
        self.config = config
        self.model_type = config['model type']
        self.image_size = image_size
        self.number_of_output_channels = number_of_output_channels
        self.num_base_filters = config["number of base filters"]
        self.kernel_size = config["convolution kernel size"]
        self.optimizer = config["optimizer"]
        self.dropout = config["dropout ratio"]
        self.patch_size = config["patch size"]
        self.projection_dim = config["projection dim"]
        self.num_attention_heads = config["num heads"]
        self.num_transformer_layers = config["transformer layers"]
        self.vit_encoder = ViTEncoder(image_size=self.image_size[0],
                                      num_input_channels=self.image_size[-1],
                                      patch_size=self.patch_size,
                                      num_layers=self.num_transformer_layers,
                                      num_heads=self.num_attention_heads,
                                      d_model=self.projection_dim)
        encoder_input_dim = self.vit_encoder.get_output_shape()
        self.encoder_input_dim = np.array([encoder_input_dim[0] // math.sqrt(encoder_input_dim[0]),
                                           encoder_input_dim[0] // math.sqrt(encoder_input_dim[0]),
                                           encoder_input_dim[1]]).astype(int)
        self.decoder = TransformerDecoder(self.projection_dim,
                                          self.number_of_output_channels,
                                          self.patch_size,
                                          encoder_input_dim[0])

    def forward(self, x):
        x = self.vit_encoder(x)
        x = self.decoder(x)
        return x


class VIT_encoder_CNN_decoder(nn.Module):
    def __init__(self, config, image_size, number_of_output_channels):
        super(VIT_encoder_CNN_decoder, self).__init__()
        self.config = config
        self.model_type = config['model type']
        self.image_size = image_size
        self.number_of_output_channels = number_of_output_channels
        self.num_base_filters = config["number of base filters"]
        self.kernel_size = config["convolution kernel size"]
        self.optimizer = config["optimizer"]
        self.dropout = config["dropout ratio"]
        self.patch_size = config["patch size"]
        self.projection_dim = config["projection dim"]
        self.num_attention_heads = config["num heads"]
        self.num_transformer_layers = config["transformer layers"]
        self.vit_encoder = CustomViT(image_size=image_size[1],
                                     patch_size=self.patch_size,
                                     dim=self.projection_dim,
                                     depth=self.num_transformer_layers,
                                     heads=self.num_attention_heads,
                                     mlp_dim=self.projection_dim * 4)
        # summary(self.vit_encoder, (4, 192, 192))
        self.cnn_decoder = CNN_Decoder(num_output_channels=self.number_of_output_channels,
                                       kernel_size=self.kernel_size,
                                       num_base_filters=self.num_base_filters,
                                       projection_dim=self.projection_dim)

    def forward(self, x):
        x = self.vit_encoder(x)
        x = self.cnn_decoder(x)
        return x


if __name__ == "__main__":
    # from vit_pytorch import ViT
    # model = ViT('B_16_imagenet1k', pretrained=True)
    # summary(model, (3, 384, 384))

    image_size = (192, 192, 4)
    x0 = torch.randn(1, 4, 192, 192)

    # vit_encoder = VIT_encoder_CNN_decoder()
    # summary(vit_encoder, x0.shape[1:])
    pass
    # y0 = vit_encoder(x0)

    # configuration_path = 'train_config.json'
    # with open(configuration_path) as C:
    #     config = json.load(C)
    # encoder_decoder = VIT_encoder_CNN_decoder(config, image_size, number_of_output_channels=10)
    # summary(encoder_decoder, x0.shape[1:])
    # pass
