import torch.nn as nn
import torch

class Unet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        
       
        self.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd1    = nn.BatchNorm2d(64)
        self.bnd1_1  = nn.BatchNorm2d(64)
        self.mp1     = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.conv2   = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd2    = nn.BatchNorm2d(128)
        self.bnd2_2  = nn.BatchNorm2d(128)
        self.mp2     = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.conv3   = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd3    = nn.BatchNorm2d(256)
        self.bnd3_3  = nn.BatchNorm2d(256)
        self.mp3     = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4   = nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv4_4   = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd4    = nn.BatchNorm2d(512)
        self.bnd4_4  = nn.BatchNorm2d(512)
        self.mp4     = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5   = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd5    = nn.BatchNorm2d(1024)
        self.mp5     = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv6   = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd6    = nn.BatchNorm2d(1024)
        
        self.relu    = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        
        self.bn1     = nn.BatchNorm2d(512)
        self.conv7   = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv7_7   = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn7     = nn.BatchNorm2d(512)
        self.bn7_7     = nn.BatchNorm2d(512)
        
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)        
        self.conv8   = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv8_8   = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn8     = nn.BatchNorm2d(256)
        self.bn8_8     = nn.BatchNorm2d(256)
        
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.conv9   = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv9_9   = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn9     = nn.BatchNorm2d(128)
        self.bn9_9     = nn.BatchNorm2d(128)
        
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.conv10   = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv10_10   = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn10     = nn.BatchNorm2d(64)
        self.bn10_10     = nn.BatchNorm2d(64)
        
        
#         self.deconv5= nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn5     = nn.BatchNorm2d(32)
#         self.conv11   = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1)
#         self.bn11     = nn.BatchNorm2d(32)
        
        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)

    def forward(self, x):
        #print(x.size())
        # Complete the forward function for the rest of the encoder
        x1 = self.bnd1(self.relu(self.conv1(x)))
        x1 = self.bnd1_1(self.relu(self.conv1_1(x1)))
        
        x2 = self.mp1(x1)
        x2 = self.bnd2(self.relu(self.conv2(x2)))
        x2 = self.bnd2_2(self.relu(self.conv2_2(x2)))
        
        x3 = self.mp2(x2)
        x3 = self.bnd3(self.relu(self.conv3(x3)))
        x3 = self.bnd3_3(self.relu(self.conv3_3(x3)))
        
        x4 = self.mp3(x3)
        x4 = self.bnd4(self.relu(self.conv4(x4)))
        x4 = self.bnd4_4(self.relu(self.conv4_4(x4)))
        
        x5 = self.mp4(x4)
        x5 = self.bnd5(self.relu(self.conv5(x5)))
        
        out_encoder = self.bnd6(self.relu(self.conv6(x5)))
        
        # Complete the forward function for the rest of the decoder
        x7 = self.bn1(self.relu(self.deconv1(out_encoder)))
        x7 = torch.cat((x7, x4), 1)
        x7 = self.bn7(self.relu(self.conv7(x7)))
        x7 = self.bn7_7(self.relu(self.conv7_7(x7)))
        
        
        x8 = self.bn2(self.relu(self.deconv2(x7)))
        x8 = torch.cat((x8, x3), dim = 1)
        x8 = self.bn8(self.relu(self.conv8(x8)))
        x8 = self.bn8_8(self.relu(self.conv8_8(x8)))
        
        x9 = self.bn3(self.relu(self.deconv3(x8)))       
        x9 = torch.cat((x9, x2), dim = 1)
        x9 = self.bn9(self.relu(self.conv9(x9)))
        x9 = self.bn9_9(self.relu(self.conv9_9(x9)))
        
        x10 = self.bn4(self.relu(self.deconv4(x9)))
        x10 = torch.cat((x10, x1), dim = 1)
        x10 = self.bn10(self.relu(self.conv10(x10)))
        out_decoder = self.bn10_10(self.relu(self.conv10_10(x10)))
        #print(out_decoder.size())
        
#         x11 = self.bn5(self.relu(self.deconv5(x10)))
#         out_decoder = self.bn11(self.relu(self.conv11(x11)))
        
        score = self.classifier(out_decoder)
        
        return score  # size=(N, n_class, x.H/1, x.W/1)