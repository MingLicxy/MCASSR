"""
The code comes from https://github.com/Adamdad/kat/blob/main/katransformer.py
Kolmogorov–Arnold Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
#TODO kat_rational已经在LIIF中安装
from kat_rational import KAT_Group

class KAN(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_cfg=dict(type="KAT", act_init=["identity", "gelu"]),
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act1 = KAT_Group(mode = act_cfg['act_init'][0])
        #self.drop1 = nn.Dropout(drop)
        self.act2 = KAT_Group(mode = act_cfg['act_init'][1])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        #self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.act1(x)
        #x = self.drop1(x)
        x = self.fc1(x)
        x = self.act2(x)
        #x = self.drop2(x)
        x = self.fc2(x)
        return x

if __name__== '__main__':
    #############Test Model Complexity #############
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis    
    
    x = torch.randn(1, 2304, 64).cuda() #[B,L,C]
    

    model = KAN(in_features=64, hidden_features=64, out_features=3).cuda()
    # model = SAFMN(dim=36, n_blocks=12, ffn_scale=2.0, upscaling_factor=2)
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)