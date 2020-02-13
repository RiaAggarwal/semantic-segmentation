import torch.nn as nn
import torchvision.models as models

class VGGTransfer(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        vgg = models.vgg16(pretrained=True)
        for param in vgg.parameters():
            param.requires_grad = False
        self.features = vgg.features
        self.relu    = nn.ReLU(inplace=True)
        #decoder
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bnd1     = nn.BatchNorm2d(512)        
        self.conv1   = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv1_1   = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.bn1_1     = nn.BatchNorm2d(512)
        
        self.deconv2 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bnd2     = nn.BatchNorm2d(512)        
        self.conv2   = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2_2   = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn2     = nn.BatchNorm2d(512)
        self.bn2_2     = nn.BatchNorm2d(512)
        
        
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bnd3     = nn.BatchNorm2d(256)        
        self.conv3   = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv3_3   = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn3     = nn.BatchNorm2d(256)
        self.bn3_3     = nn.BatchNorm2d(256)
        
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bnd4     = nn.BatchNorm2d(128)
        self.conv4   = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv4_4   = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn4     = nn.BatchNorm2d(128)
        self.bn4_4     = nn.BatchNorm2d(128)
        
        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bnd5     = nn.BatchNorm2d(64)
        self.conv5   = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv5_5   = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn5     = nn.BatchNorm2d(64)
        self.bn5_5     = nn.BatchNorm2d(64)
        
        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)

    def forward(self, x):
#         print('in forward', x.size())
        out_encoder = self.features(x)
        
        
#         print('out encoder', out_encoder.size())
        x1 = self.bnd1(self.relu(self.deconv1(out_encoder)))
        x1 = self.bn1(self.relu(self.conv1(x1)))
        x1 = self.bn1_1(self.relu(self.conv1_1(x1)))
        
#         print('x1', x1.size())
        x2 = self.bnd2(self.relu(self.deconv2(x1)))
        x2 = self.bn2(self.relu(self.conv2(x2)))
        x2 = self.bn2_2(self.relu(self.conv2_2(x2)))

#         print('x2', x2.size())
        x3 = self.bnd3(self.relu(self.deconv3(x2)))
        x3 = self.bn3(self.relu(self.conv3(x3)))
        x3 = self.bn3_3(self.relu(self.conv3_3(x3)))

#         print('x3', x3.size())
        x4 = self.bnd4(self.relu(self.deconv4(x3)))
        x4 = self.bn4(self.relu(self.conv4(x4)))
        x4 = self.bn4_4(self.relu(self.conv4_4(x4)))

#         print('x4', x4.size())
        x5 = self.bnd5(self.relu(self.deconv5(x4)))
        x5 = self.bn5(self.relu(self.conv5(x5)))
        out_decoder = self.bn5_5(self.relu(self.conv5_5(x5)))

#         print('out_decoder', out_decoder.size())
     
        score = self.classifier(out_decoder)
        
        return score  # size=(N, n_class, x.H/1, x.W/1)