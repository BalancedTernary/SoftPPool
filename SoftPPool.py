import torch

class SoftPPool(torch.nn.Module):
    def __init__(self,**kwargs):
        super(SoftPPool, self).__init__()
        self.Pool=torch.nn.AvgPool2d(**kwargs)

    def forward(self,input):
        output=(input.cosh()-1)*input.sign()
        output=self.Pool(output)
        output=output.asinh()
        return output

            
