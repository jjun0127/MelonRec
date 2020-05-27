import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, D_in, H):
        super(Encoder,self).__init__()
        self.layer = nn.Sequential(
                        nn.Linear(D_in, H),
                        nn.ReLU())
        
                
    def forward(self,x):
        out = self.layer(x)
        return out
    
# class DenoisingEncoder(nn.Module):
#     def __init__(self, D_in, H):
#         super(Encoder,self).__init__()
#         self.layer = nn.Sequential(
#                         nn.Linear(D_in, H),
#                         nn.ReLU())
#
#
#     def forward(self,x):
#         size = x.size()
#         out = self.layer(x)
#         out = out.view(size[0], -1)
#         return out
    
class Decoder(nn.Module):
    def __init__(self, D_out, H):
        super(Decoder, self).__init__()
        self.layer = nn.Sequential(
                        nn.Linear(H, D_out),
                        nn.Sigmoid())
        
    def forward(self,x):
        out = self.layer(x)
        return out

