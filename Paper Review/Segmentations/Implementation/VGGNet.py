import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary as model_summary

ranges = {'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))}

class VGGNet(nn.Module):
    def __init__(self, pretrained=True):
        super(VGGNet, self).__init__()
        self.ranges = ranges['vgg16']
        self.features = models.vgg16(weights=pretrained).features

    def forward(self, x):
        output = {}
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x
        return output

    
class FCNs(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        
        x5 = output['x5']  
        x4 = output['x4']  
        x3 = output['x3']  
        x2 = output['x2'] 
        x1 = output['x1']  

        score = self.bn1(self.relu(self.deconv1(x5)))     
        score = score + x4                            
        score = self.bn2(self.relu(self.deconv2(score)))
        score = score + x3                            
        score = self.bn3(self.relu(self.deconv3(score))) 
        score = score + x2                              
        score = self.bn4(self.relu(self.deconv4(score)))  
        score = score + x1                                
        score = self.bn5(self.relu(self.deconv5(score)))  
        score = self.classifier(score)                    
        
        return score  
    
vgg16 = VGGNet(pretrained=True)
model = FCNs(vgg16, 2)
model