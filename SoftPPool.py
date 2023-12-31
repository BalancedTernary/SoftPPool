import torch

class SoftPPool2D(torch.nn.Module):
    def __init__(self,**kwargs):
        super(SoftPPool2D, self).__init__()
        self.Pool=torch.nn.AvgPool2d(**kwargs)

    def forward(self,input):
        output=(input.cosh()-1)*input.sign()
        output=self.Pool(output)
        output=output.asinh()
        return output

            
