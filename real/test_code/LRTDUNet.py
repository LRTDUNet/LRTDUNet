
import torch.nn as nn

import torch.nn.functional as F
from einops import rearrange
import math
import warnings

from torch.nn.init import _calculate_fan_in_and_fan_out
# from srfm import  *


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, enc=None,dec=None,*args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


###############


import torch
import torch.nn as nn


#############################3
class DSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            dir="none"
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        # self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.temperature=nn.Parameter(torch.ones(1,1,1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.dir=dir
        if self.dir=="v":
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=dim * 3,
                                    bias=False)
        elif self.dir=='h':
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=dim * 3,
                                         bias=False)
        elif self.dir=='hv':
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=5, stride=1, padding=2, groups=dim * 3, bias=False)
        elif self.dir == 'hvv':
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=7, stride=1, padding=3, groups=dim * 3,
                                        bias=False)

        else:
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.mask_generator = nn.Dropout(0.2)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.pos_emb = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
                                 GELU(),nn.Conv2d(dim,dim,3,1,1,bias=False,groups=dim)
        )
        self.dim = dim
    def forward(self, x_in,enc=None,dec=None):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x_out=torch.permute(x_in,(0,3,1,2))

        qkv=self.qkv_dwconv(self.qkv(x_out))
        q, k, v = qkv.chunk(3, dim=1)
        v_in = v
        q = rearrange(q, 'b (head c) h w -> b (head c) (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b (head c) (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b (head c) (h w)', head=self.num_heads)

        q= torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1))*self.temperature

        mask = self.mask_generator(torch.ones_like(attn))
        attn = attn * mask  # (1,1,16384,16384)
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out1 = rearrange(out, 'b (head c) (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)#(1,31,128,128)
        out_p = self.pos_emb(v_in).permute(0, 2, 3, 1)
        out1 = self.project_out(out1)
        out1=torch.permute(out1,(0,2,3,1))+out_p #(1,128,128,31)
        return out1

######################
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x,enc=None,dec=None):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class MSABS(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
            dir="NONE",
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.dir=dir
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                DSAB(dim=dim, dim_head=dim_head, heads=heads,dir=self.dir),
                PreNorm(dim, FeedForward(dim=dim))
            ]))
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
                x = attn(x) + x
                x = ff(x) + x
        out = x.permute(0, 3, 1, 2)

        return out




class Unet(nn.Module):
    def __init__(self, in_nc=31):
        super(Unet, self).__init__()
        self.down1=nn.Conv2d(in_nc,in_nc*2,4,2,1)
        self.cat_conv1=nn.Conv2d(2*in_nc,in_nc,1)
        self.down2 = nn.Conv2d(2*in_nc, in_nc* 4, 4, 2, 1)
        self.cat_conv2 = nn.Conv2d(4 * in_nc, 2*in_nc, 1)
        self.up2=nn.ConvTranspose2d(4*in_nc,2*in_nc,2,2)
        self.up1=nn.ConvTranspose2d(2* in_nc, in_nc, 2, 2)
        self.mapping=nn.Conv2d(in_nc,in_nc,3,1,1)

    def forward(self,x):
        x1=self.down1(x)
        x2=self.down2(x1)

        x1_out=self.cat_conv2(torch.cat([x1,self.up2(x2)],dim=1))
        x_out=self.cat_conv1(torch.cat([x,self.up1(x1_out)],dim=1))
        out=self.mapping(x_out)
        return out
#



class HyPaNet(nn.Module):
    def __init__(self, in_nc=31, out_nc=1, channel=31):
        super(HyPaNet, self).__init__()
        self.unet=Unet(in_nc)
        self.mlp = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                # nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus()
        )
        # self.relu = nn.ReLU(inplace=True)
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.out_nc = out_nc

    def forward(self, x):
        x=self.unet(x)
        # x0=x
        # x = self.down_sample(self.relu(self.fution(x)))
        # x = self.avg_pool(x)
        x = self.mlp(x) + 1e-6
        # x = self.avg_pool(x)
        # x = self.mlp(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)




class DMST(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, stage=2, num_blocks=[1,1,1],dir="o"):
        super(DMST, self).__init__()
        self.dim = dim
        self.stage = stage
        self.dir=dir
        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
        # self.embedding=nn.Conv2d(in_dim,self.dim,)
        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        # dim_stage1= dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSABS(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim, dir=self.dir),
                # MSAB(
                #     dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),

                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),#downsize
            ]))
            dim_stage *= 2


        # Bottleneck
        self.bottleneck =nn.Sequential(
            MSABS(
                    dim=dim_stage, num_blocks=num_blocks[-1], dim_head=dim, heads=dim_stage // dim,dir=self.dir),
            # MSAB(
            # dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1]),

        )

        # Decoder
        self.decoder_layers = nn.ModuleList([])

        # self.decoder_layers1 = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSABS(
                    dim=dim_stage//2, num_blocks=num_blocks[i], dim_head=dim, heads=(dim_stage//2) // dim),


            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x) #[1,31,256,256]
        fea1=fea
        # Encoder
        fea_encoder = []

        for (MSAB, FeaDownSample) in self.encoder_layers:

            fea = MSAB(fea)

            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        fea = self.bottleneck(fea)#level 2 124. level 3  248 level1 62


        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = LeWinBlcok(fea)

        out=self.mapping(fea)+x
        return out






from srdm import  *

class MGUHST(nn.Module):

    def __init__(self, num_iterations=4):
        super(MGUHST, self).__init__()
        self.para_estimator = HyPaNet(in_nc=31, out_nc=31)
        self.conv_in= nn.Conv2d(3, 31, 3, padding=1, bias=True)
        self.num_iterations = num_iterations
        self.denoisers = nn.ModuleList([])

        # dir_list=["h","v","hv","hvv","o"]
        # hw_list=[1,1,1,1,1]
        # self.conv_out=nn.Conv2d(31,31,3,1,1)
        for i in range(num_iterations):
            self.denoisers.append(
                 nn.ModuleList([BMST(31, 31, 31, 2, [1, 1, 3]),
                # nn.ModuleList([DMST(31, 31, 31, 2, [1, 1, 4]),
                HyPaNet(31,31)])

            )
    def initial(self, x):
        """
        """
        # x=self.conv_before(x)
        z = self.conv_in(x)
        alpha = self.para_estimator(z)#[1,1,1,1],[1,1,1,1]
        return z, alpha
    def forward(self,x):
        res=[]

        b, c, h_inp, w_inp = x.shape
        hb, wb = 32, 32
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        z, alpha = self.initial(x)
        z0=z
        for i in range(self.num_iterations):
            z = z0*alpha + z
            z=self.denoisers[i][0](z)
            alpha=self.denoisers[i][1](z)
            res.append(z[:,:,:h_inp,:w_inp])
        return res
        # z=z+z0
        # return z[:, :, :h_inp, :w_inp]


model=MGUHST()
# # # # # # from  self_utils import *
# # # # # # # criterion_wave=Loss_Wavelet()
# # # # # # # criterion_mrae = Loss_MRAE()
# # # # # # # # list_weight=[0.2,0.3,0.4,0.5,0.6]
a=torch.ones(1,3,256,256)
# # # # # # # labels=torch.ones(1,31,256,256)
from thop import profile
flops, params = profile(model, inputs=(a,))
total = sum([param.nelement() for param in model.parameters()])
print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
print("FLOPs=", str(flops / 1e6) + '{}'.format("M"))
print("Number of parameters: %.2fM" % (total / 1e6))