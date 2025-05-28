import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, einsum
from torch.nn import functional as F
from Nets.DAMT import DAMTransformer
from Nets.DACF import DACF


# BR
class Basic_Residual_Module(nn.Module):
    def __init__(self, input_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(48, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = x + residual
        x = self.relu(x)
        return x

# ASPP
class Atrous_Spatial_Pyramid_Pooling_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.Initial = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.dilatation_conv_1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.dilatation_conv_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=2, stride=1, dilation=2),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.dilatation_conv_3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=3, stride=1, dilation=3),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.dilatation_conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=4, stride=1, dilation=4),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.dilatation_conv_5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.Initial(x)
        x1 = self.dilatation_conv_1(x)
        x2 = self.dilatation_conv_2(x)
        x3 = self.dilatation_conv_3(x)
        x4 = self.dilatation_conv_4(x)
        concatenation = x1 + x2 + x3 + x4
        x5 = self.dilatation_conv_5(concatenation)
        x = x5 + residual
        x = self.relu(x)
        return x


class DM_LayerNorm(nn.Module):  # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class DM_conv3x3(nn.Module):
    "3x3 convolution with padding"

    def __init__(self, input_dim, output_dim, stride=1):
        super().__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_dim, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv3x3(x)
        return x


# DAMTM
class Dual_Attention_Mixing_Transformer_Module(nn.Module):
    def __init__(self, input_dim=32, dropout=0.):
        super().__init__()
        self.CvT = nn.Sequential(

            nn.Conv2d(32, 48, kernel_size=7, stride=4, padding=3),
            DM_LayerNorm(48),
            DAMTransformer(dim=48),

            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            DM_LayerNorm(64),
            DAMTransformer(dim=64),
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv3x3_1 = DM_conv3x3(input_dim=64, output_dim=48)
        self.conv3x3_2 = DM_conv3x3(input_dim=48, output_dim=32)

    def forward(self, x):
        residual = x
        x = self.CvT(x)
        x = F.interpolate(x, mode='bilinear', size=(x.shape[2] * 2, x.shape[3] * 2))
        x = self.conv3x3_1(x)
        x = F.interpolate(x, mode='bilinear', size=(residual.shape[2], residual.shape[3]))
        x = self.conv3x3_2(x)
        x = x + residual
        x = self.relu(x)
        return x


class CT_LayerNorm(nn.Module):  # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class CT_PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = CT_LayerNorm(dim)
        self.fn = fn

    def forward(self, x, y, **kwargs):
        x = self.norm(x)
        y = self.norm(y)
        return self.fn(x, y, **kwargs)


class CT_DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class CT_Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = CT_DepthWiseConv2d(dim, inner_dim, proj_kernel, padding=padding, stride=1, bias=False)
        self.to_kv = CT_DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding=padding, stride=kv_proj_stride,
                                        bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x, y):
        shapex = x.shape
        bx, nx, _x, wx, hx = *shapex, self.heads
        qx = self.to_q(x)
        kx, vx = self.to_kv(x).chunk(2, dim=1)
        qx, kx, vx = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=hx), (qx, kx, vx))
        shapey = y.shape
        by, ny, _y, wy, hy = *shapey, self.heads
        qy = self.to_q(y)
        ky, vy = self.to_kv(y).chunk(2, dim=1)
        qy, ky, vy = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=hy), (qy, ky, vy))

        dotsx = einsum('b i d, b j d -> b i j', qx, ky) * self.scale
        attnx = self.attend(dotsx)
        attnx = self.dropout(attnx)
        outx = einsum('b i j, b j d -> b i d', attnx, vy)
        outx = rearrange(outx, '(b h) (x y) d -> b (h d) x y', h=hx, y=wx)

        dotsy = einsum('b i d, b j d -> b i j', qy, kx) * self.scale
        attny = self.attend(dotsy)
        attny = self.dropout(attny)
        outy = einsum('b i j, b j d -> b i d', attny, vx)
        outy = rearrange(outy, '(b h) (x y) d -> b (h d) x y', h=hy, y=wy)

        # out = torch.cat([outx, outy], dim=1)

        return self.to_out(outx), self.to_out(outy)


class CT_FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x, y):
        return self.net(x), self.net(y)


# CTIT
class CT_Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head=64, mlp_mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CT_PreNorm(dim, CT_Attention(dim, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride, heads=heads,
                                             dim_head=dim_head, dropout=dropout)),
                CT_PreNorm(dim, CT_FeedForward(dim, mlp_mult, dropout=dropout))
            ]))

    def forward(self, x, y):
        for attn, ff in self.layers:
            x1, y1 = attn(x, y)
            x2 = x1 + x
            y2 = y1 + y
            x3, y3 = ff(x2, y2)
            x4 = x3 + x2
            y4 = y3 + y2
        return x4, y4


class CT_conv3x3(nn.Module):
    "3x3 convolution with padding"

    def __init__(self, input_dim, output_dim, stride=1):
        super().__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_dim, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv3x3(x)
        return x


# CTITM
class Cross_Task_Interaction_Transformer_Module(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=7, stride=4, padding=3),
            CT_LayerNorm(48),
        )
        self.cdit = CT_Transformer(dim=48, proj_kernel=3, kv_proj_stride=4, heads=1, depth=1, mlp_mult=2,
                                   dropout=dropout)
        # self.down2 = nn.Sequential(
        #     nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
        #     CC_LayerNorm(64),
        # )
        # self.cdtb2 = CD_Transformer_Block(dim=64, proj_kernel=3, kv_proj_stride=2, heads=2, depth=1, mlp_mult=4,
        #                          dropout=dropout)

        self.relu = nn.ReLU(inplace=True)
        # self.conv3x3_1 = CD_conv3x3(input_dim=64, output_dim=48)
        self.conv3x3_2 = CT_conv3x3(input_dim=48, output_dim=32)

    def forward(self, x, y):
        residualx = x
        residualy = y

        x, y = self.cdit(self.down1(x), self.down1(y))
        # x, y = self.cdtb2(self.down2(x), self.down2(y))

        # x = F.interpolate(x, mode='bilinear', size=(x.shape[2] * 2, x.shape[3] * 2))
        # x = self.conv3x3_1(x)
        x = F.interpolate(x, mode='bilinear', size=(residualx.shape[2], residualx.shape[3]))
        x = self.conv3x3_2(x)
        x = x + residualx
        x = self.relu(x)

        # y = F.interpolate(y, mode='bilinear', size=(y.shape[2] * 2, y.shape[3] * 2))
        # y = self.conv3x3_1(y)
        y = F.interpolate(y, mode='bilinear', size=(residualy.shape[2], residualy.shape[3]))
        y = self.conv3x3_2(y)
        y = y + residualy
        y = self.relu(y)
        return x, y


class CSI_DMT(nn.Module):
    def __init__(
            self,
            *,
            img_channels=3,
            dropout=0.
    ):
        super().__init__()

        # Shared feature extraction
        self.sfe = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            Atrous_Spatial_Pyramid_Pooling_Module(),
            Dual_Attention_Mixing_Transformer_Module(),
        )

        # Feature fusion
        self.mixer = nn.Sequential(
            nn.Conv2d(32 * 2, 32, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
            Basic_Residual_Module(),
        )

        # Fused image task
        self.csva_g_1 = Dual_Attention_Mixing_Transformer_Module()
        self.csva_g_2 = Dual_Attention_Mixing_Transformer_Module()
        self.br_g = Basic_Residual_Module()
        self.outconv_g = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid(),
        )

        # Decision map task
        self.csva_d_1 = Dual_Attention_Mixing_Transformer_Module()
        self.br_d = Basic_Residual_Module()
        self.outconv_d = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid(),
        )

        # Cross-Task Semantic Interaction
        self.cdc1 = Cross_Task_Interaction_Transformer_Module()
        self.cdc2 = Cross_Task_Interaction_Transformer_Module()
        self.cafm1 = DACF()
        self.cafm2 = DACF()


    def forward(self, A, B):
        Feature_A = self.sfe(A)
        Feature_B = self.sfe(B)

        concatenation = torch.cat([Feature_A, Feature_B], dim=1)
        F = self.mixer(concatenation)

        FG1 = self.csva_g_1(F)
        FD1 = self.csva_d_1(F)

        x1, y1 = self.cdc1(FG1, FD1)
        x2, y2 = self.cafm1(FG1, FD1)

        FG1 = x1 + x2
        FD1 = y1 + y2

        FG2 = self.csva_g_2(FG1)

        FG3 = self.br_g(FG2)
        FD2 = self.br_d(FD1)

        x3, y3 = self.cdc2(FG3, FD2)
        x4, y4 = self.cafm2(FG3, FD2)

        FG3 = x3 + x4
        FD2 = y3 + y4

        FGOut = self.outconv_g(FG3)
        FDOut = self.outconv_d(FD2)

        return FGOut, FDOut




if __name__ == '__main__':
    test_tensor_A = torch.zeros((1, 3, 111, 111)).to('cuda')
    test_tensor_B = torch.rand((1, 3, 111, 111)).to('cuda')
    model = CSI_DMT().to('cuda')
    # num_params = 0
    # for p in model.parameters():
    #     num_params += p.numel()
    # print(model)
    # print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
    FG, FD = model(test_tensor_A, test_tensor_B)
    print(FG.shape)
    print(FD.shape)
