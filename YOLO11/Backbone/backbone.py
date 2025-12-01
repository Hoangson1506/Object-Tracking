import torch
import torch.nn as nn
from torch.nn.modules import conv

#Tự động tính giá trị padding để in=out
def cal_pad(ker, pad = None, dil = 1):
    if (dil>1):
        ker = dil *(ker - 1) + 1 if isinstance(ker,int) else [dil * (x - 1) for x in ker]
    if pad is None:
        pad = ker // 2 if isinstance(ker,int) else [x // 2 for x in ker]

class Conv(nn.Module):
    def __init__(self, in_chan, out_chan, activation, ker = 3, pad = 1, stride = 1 ):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, ker, stride, pad, bias = False)
        self.norm = nn.BatchNorm2d(out_chan, eps = 1e-3, momen = 0.03)
        self.activation = activation

    def forward(self, x, fuse = False):
        if(fuse is False):
            return self.activation(self.norm(self.conv(x)))
        else:
            return self.activation(self.conv(x))

class Residual(nn.Module):
    def __init__(self, in_chan,e = 2):
        super().__init__()
        self.conv1 = Conv(in_chan, in_chan // e, nn.SiLU())
        self.conv2 = Conv(in_chan // e, in_chan, nn.SiLU())
    
    def forward(self, x):
        return(x + self.conv2(self.conv1(x)))

class C3K(nn.Module):
    def __init__(self, in_chan, out_chan,n):
        super().__init__()
        self.conv1 = Conv(in_chan, out_chan//2, nn.SiLU())
        self.conv2 = Conv(in_chan, out_chan//2, nn.SiLU())
        self.conv3 = Conv(2 * (out_chan // 2), out_chan,nn.SiLU())
        self.BoNe  = nn.Sequential()
        for i in range(n):
            self.BoNe.add_module(f"res_{i}", Residual(out_chan, e = 1))

    def forward(self, x):
        x_1 = self.BoNe(self.conv1(x_1))
        return(self.conv3(torch.concat(x_1, self.conv2(x),dim = 1)))

class C3K2(nn.Module):
    def __init__(self, in_chan, out_chan, n, csp):
        super().__init__()
        self.conv1 = Conv(in_chan, 2 * (in_chan // 2), nn.SiLU())
        self.conv2 = Conv((2 + n) * (in_chan // 2), out_chan, nn.SiLU())
        self.ShortCut = nn.ModuleList()
        if csp:
            for i in range(n):
                self.ShortCut.add_module(f"res_{i}", Residual(out_chan // 2, e = 1))
        else:
            for i in range(n):
                self.ShortCut.add_module(f"res_{i}", C3K(out_chan // 2, out_chan // 2))
        
    def forward(self, x):
        x_1 = torch.chunk(self.conv1(x) , 2, 1)
        y = list(x_1)
        for i in range(y[-1].size(1)):
            y.append()

a = torch.rand(1,2,3)
print(a.size(1))