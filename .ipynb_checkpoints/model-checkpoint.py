from torch.nn import *
from act import WiBReLU
import torch

class VanillaModel(Module):
    
    def __init__(self, in_chs, out_chs, ks, s, p, n_cls, in_fs, act = "wib"):
        super().__init__()
        
        assert act in ["regular", "wib", "leaky", "prelu", "gelu"], "Please choose proper activation function from the list!"
        self.conv_1 = Conv2d(in_channels = in_chs, out_channels = out_chs, kernel_size = ks, stride = s, padding = p)
        self.act = ReLU() if act == "regular" else (WiBReLU() if act == "wib" else (LeakyReLU() if act == "leaky" else (PReLU() if act == "prelu" else GELU())) )
        self.mp = MaxPool2d(kernel_size = 2, stride = 2)
        self.conv_2 = Conv2d(in_channels = out_chs, out_channels = out_chs * 2, kernel_size = ks, stride = s, padding = p)
        self.conv_3 = Conv2d(in_channels = out_chs * 2, out_channels = out_chs * 4, kernel_size = ks, stride = s, padding = p)
        self.conv_4 = Conv2d(in_channels = out_chs * 4, out_channels = out_chs * 8, kernel_size = ks, stride = s, padding = p)
        self.linear_1 = Linear(in_features = in_fs, out_features = in_fs // 2) 
        self.linear_2 = Linear(in_features = in_fs // 2, out_features = n_cls)
        
    def forward(self, inp): 
        
        out = self.conv_1(inp)
        # print(f"first: {torch.mean(out)}")
        out = self.act(out) 
        out = self.mp(out) 
        
        out = self.conv_2(out)
        # print(f"second: {torch.mean(out)}")
        out = self.act(out) 
        out = self.mp(out) 
        
        out = self.conv_3(out)
        # print(f"third: {torch.mean(out)}")
        out = self.act(out) 
        out = self.mp(out) 
        
        out = self.conv_4(out)
        # print(f"fourth: {torch.mean(out)}")
        out = self.act(out) 
        out = self.mp(out) 
        
        bs = out.shape[0]
        out = out.view(bs, -1)
        # print(f"fifth: {torch.mean(out)}")
        
        out = self.act(self.linear_1(out))
        out = self.linear_2(out)
         
        return out