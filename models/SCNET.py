import torch
import torch.nn as nn
from transformer import BasicLayer, PatchMerging, PatchEmbed

def show_fp(fp):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fp_img = fp[0,...].max(dim=0)[0].detach().cpu()

    sns.heatmap(fp_img, cmap='rainbow', cbar=True)
    plt.show()



class SCNET(nn.Module):
    def __init__(self, channels, num_of_layers=12,
                pretrain_img_size=224,
                patch_size=1,
                in_chans=32,
                embed_dim=32,
                depths=[1, 1],
                num_heads=[1, 1],
                window_size=[50, 50],
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                norm_layer=nn.LayerNorm,
                ape=False,
                patch_norm=True,
                out_indices=[1],
                frozen_stages=-1,
                use_checkpoint=False):
        super(SCNET, self).__init__()
        kernel_size = 3
        padding = 1
        self.dim = 32
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=self.dim, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(self.dim))
            layers.append(nn.ReLU(inplace=True))
        self.dncnn = nn.Sequential(*layers)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.patch_norm = patch_norm
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=1, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)


        # build layers
        self.num_layers = len(depths)
        self.trans = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint)
            self.trans.append(layer)

        # add a norm layer for each output
        self.out_indices = out_indices
        for i_layer in out_indices:
            layer = norm_layer(self.dim)
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)


        # build former layer
        self.in_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.dim, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
        )

        # build out layer
        self.out_layer = nn.Sequential(
            # nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Conv2d(in_channels=self.dim, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.ReLU(),
        )

    def forward(self, x):
        # x1 = self.dncnn(x) # [1, 64, 100, 100]
        # x1_ = x1

        # # show_fp(x1)
        # x1=x

        # x1 = self.patch_embed(x) # [1, 64, 100, 100]
        x1 = self.in_layer(x) 
        x1_out = x1
        Wh, Ww = x1.size(2), x1.size(3)
        x1 = x1.flatten(2).transpose(1, 2) # [1, 10000, 64]
        for i in range(self.num_layers):
            layer = self.trans[i]
            x_out, H, W, x1, Wh, Ww = layer(x1, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                x2 = x_out.view(-1, H, W, self.dim).permute(0, 3, 1, 2).contiguous() # 


        # show_fp(x2)
        # x3 = self.dncnn(x2) # [1, 64, 100, 100]
        out = self.out_layer(x2) # [1, 1, 100, 100]
        return out, x1_out, x2
