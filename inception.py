import torch.nn as nn
import torch

class Inception(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        
        self.conv1_left   = nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0, dilation=1)
        self.mp1_1_left   = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv1_3_1   = nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv1_3_3   = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, dilation=1)
        self.mp1_3       = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1_5_1   = nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv1_5_5   = nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2, dilation=1)
        self.mp1_5       = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.mp1_1_right   = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1_right   = nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0, dilation=1)
        
#         Concat!!!!
        self.bne1        = nn.BatchNorm2d(32)
        self.ReLU        = nn.ReLU(inplace=True)
        
        
        self.conv2_left   = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, dilation=1)
        self.mp2_1_left   = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2_3_1   = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv2_3_3   = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, dilation=1)
        self.mp2_3       = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_5_1   = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv2_5_5   = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2, dilation=1)
        self.mp2_5       = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.mp2_1_right   = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_right   = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, dilation=1)
        
#         Concat!!!!
        self.bne2        = nn.BatchNorm2d(64)
    
        self.conv3_left   = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, dilation=1)
        self.mp3_1_left   = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv3_3_1   = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv3_3_3   = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.mp3_3       = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_5_1   = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv3_5_5   = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, dilation=1)
        self.mp3_5       = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.mp3_1_right   = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_right   = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, dilation=1)
        
#         Concat!!!!
        self.bne3        = nn.BatchNorm2d(128)
    
    
        self.conv4_left   = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, dilation=1)
        self.mp4_1_left   = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv4_3_1   = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv4_3_3   = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.mp4_3       = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_5_1   = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv4_5_5   = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, dilation=1)
        self.mp4_5       = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.mp4_1_right   = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_right   = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, dilation=1)
        
#         Concat!!!!
        self.bne4        = nn.BatchNorm2d(256)
    
    
        self.conv5_left   = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, dilation=1)
        self.mp5_1_left   = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv5_3_1   = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv5_3_3   = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.mp5_3       = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5_5_1   = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv5_5_5   = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, dilation=1)
        self.mp5_5       = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.mp5_1_right   = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_right   = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, dilation=1)
        
#         Concat!!!!
        self.bne5        = nn.BatchNorm2d(512)
        
        #####################################
        
        self.Dconv5_left   = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, dilation=1)
        self.Dmp5_1_left   = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        
        self.Dconv5_3_1   = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, dilation=1)
        self.Dconv5_3_3   = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.Dmp5_3       = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        
        self.Dconv5_5_1  = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, dilation=1)
        self.Dconv5_5_5  = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, dilation=1)
        self.Dmp5_5      = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        
        self.Dmp5_1_right   = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.Dconv5_right   = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, dilation=1)
        
        self.bnb5         =  nn.BatchNorm2d(512)
        
        
        self.Dconv4_left   = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, dilation=1)
        self.Dmp4_1_left   = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        
        self.Dconv4_3_1   = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, dilation=1)
        self.Dconv4_3_3   = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.Dmp4_3       = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        
        self.Dconv4_5_1  = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, dilation=1)
        self.Dconv4_5_5  = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, dilation=1)
        self.Dmp4_5      = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        
        self.Dmp4_1_right   = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.Dconv4_right   = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, dilation=1)
        
        self.bnb4         =  nn.BatchNorm2d(256)
        
        
        self.Dconv3_left   = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0, dilation=1)
        self.Dmp3_1_left   = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        
        self.Dconv3_3_1   = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0, dilation=1)
        self.Dconv3_3_3   = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.Dmp3_3       = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        
        self.Dconv3_5_1  = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0, dilation=1)
        self.Dconv3_5_5  = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, dilation=1)
        self.Dmp3_5      = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        
        self.Dmp3_1_right   = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.Dconv3_right   = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0, dilation=1)
        
        self.bnb3         =  nn.BatchNorm2d(128)
        
        
        self.Dconv2_left   = nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, dilation=1)
        self.Dmp2_1_left   = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        
        self.Dconv2_3_1   = nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, dilation=1)
        self.Dconv2_3_3   = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, dilation=1)
        self.Dmp2_3       = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        
        self.Dconv2_5_1  = nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, dilation=1)
        self.Dconv2_5_5  = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2, dilation=1)
        self.Dmp2_5      = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        
        self.Dmp2_1_right   = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.Dconv2_right   = nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, dilation=1)
        
        self.bnb2         =  nn.BatchNorm2d(64)
        
        
        self.Dconv1_left   = nn.Conv2d(64, 8, kernel_size=1, stride=1, padding=0, dilation=1)
        self.Dmp1_1_left   = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        
        self.Dconv1_3_1   = nn.Conv2d(64, 8, kernel_size=1, stride=1, padding=0, dilation=1)
        self.Dconv1_3_3   = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, dilation=1)
        self.Dmp1_3       = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        
        self.Dconv1_5_1  = nn.Conv2d(64, 8, kernel_size=1, stride=1, padding=0, dilation=1)
        self.Dconv1_5_5  = nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2, dilation=1)
        self.Dmp1_5      = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        
        self.Dmp1_1_right   = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.Dconv1_right   = nn.Conv2d(64, 8, kernel_size=1, stride=1, padding=0, dilation=1)
        
        self.bnb1        =  nn.BatchNorm2d(32)
        
        self.classifier  = nn.Conv2d(32, self.n_class, kernel_size=1)
        


    def forward(self, x):
        x1_1 = self.mp1_1_left(self.conv1_left(x))
        x1_2 = self.mp1_3(self.conv1_3_3(self.conv1_3_1(x)))
        x1_3 = self.mp1_5(self.conv1_5_5(self.conv1_5_1(x)))
        x1_4 = self.conv1_right(self.mp1_1_right(x))

        x1_con = torch.cat((x1_1, x1_2, x1_3, x1_4), dim=1)
        x1 = self.ReLU(self.bne1(x1_con))

        x2_1 = self.mp2_1_left(self.conv2_left(x1))
        x2_2 = self.mp2_3(self.conv2_3_3(self.conv2_3_1(x1)))
        x2_3 = self.mp2_5(self.conv2_5_5(self.conv2_5_1(x1)))
        x2_4 = self.conv2_right(self.mp2_1_right(x1))

        x2_con = torch.cat((x2_1, x2_2, x2_3, x2_4), dim=1)
        x2 = self.ReLU(self.bne2(x2_con))

        x3_1 = self.mp3_1_left(self.conv3_left(x2))
        x3_2 = self.mp3_3(self.conv3_3_3(self.conv3_3_1(x2)))
        x3_3 = self.mp3_5(self.conv3_5_5(self.conv3_5_1(x2)))
        x3_4 = self.conv3_right(self.mp3_1_right(x2))

        x3_con = torch.cat((x3_1, x3_2, x3_3, x3_4), dim=1)
        x3 = self.ReLU(self.bne3(x3_con))


        x4_1 = self.mp4_1_left(self.conv4_left(x3))
        x4_2 = self.mp4_3(self.conv4_3_3(self.conv4_3_1(x3)))
        x4_3 = self.mp4_5(self.conv4_5_5(self.conv4_5_1(x3)))
        x4_4 = self.conv4_right(self.mp4_1_right(x3))

        x4_con = torch.cat((x4_1, x4_2, x4_3, x4_4), dim=1)
        x4 = self.ReLU(self.bne4(x4_con))

        x5_1 = self.mp5_1_left(self.conv5_left(x4))
        x5_2 = self.mp5_3(self.conv5_3_3(self.conv5_3_1(x4)))
        x5_3 = self.mp5_5(self.conv5_5_5(self.conv5_5_1(x4)))
        x5_4 = self.conv5_right(self.mp5_1_right(x4))

        x5_con = torch.cat((x5_1, x5_2, x5_3, x5_4), dim=1)
        out_encoder = self.ReLU(self.bne5(x5_con))


#         ---------------------------------
        xd5_1 = self.Dmp5_1_left(self.Dconv5_left(out_encoder))
        xd5_2 = self.Dmp5_3(self.Dconv5_3_3(self.Dconv5_3_1(out_encoder)))
        xd5_3 = self.Dmp5_5(self.Dconv5_5_5(self.Dconv5_5_1(out_encoder)))
        xd5_4 = self.Dconv5_right(self.Dmp5_1_right(out_encoder))
        xd5_con = torch.cat((xd5_1, xd5_2, xd5_3, xd5_4), dim=1)
        xd5 = self.ReLU(self.bnb5(xd5_con))


        xd4_1 = self.Dmp4_1_left(self.Dconv4_left(xd5))
        xd4_2 = self.Dmp4_3(self.Dconv4_3_3(self.Dconv4_3_1(xd5)))
        xd4_3 = self.Dmp4_5(self.Dconv4_5_5(self.Dconv4_5_1(xd5)))
        xd4_4 = self.Dconv4_right(self.Dmp4_1_right(xd5))
        xd4_con = torch.cat((xd4_1, xd4_2, xd4_3, xd4_4), dim=1)
        xd4 = self.ReLU(self.bnb4(xd4_con))

        xd3_1 = self.Dmp3_1_left(self.Dconv3_left(xd4))
        xd3_2 = self.Dmp3_3(self.Dconv3_3_3(self.Dconv3_3_1(xd4)))
        xd3_3 = self.Dmp3_5(self.Dconv3_5_5(self.Dconv3_5_1(xd4)))
        xd3_4 = self.Dconv3_right(self.Dmp3_1_right(xd4))
        xd3_con = torch.cat((xd3_1, xd3_2, xd3_3, xd3_4), dim=1)
        xd3 = self.ReLU(self.bnb3(xd3_con))

        xd2_1 = self.Dmp2_1_left(self.Dconv2_left(xd3))
        xd2_2 = self.Dmp2_3(self.Dconv2_3_3(self.Dconv2_3_1(xd3)))
        xd2_3 = self.Dmp2_5(self.Dconv2_5_5(self.Dconv2_5_1(xd3)))
        xd2_4 = self.Dconv2_right(self.Dmp2_1_right(xd3))
        xd2_con = torch.cat((xd2_1, xd2_2, xd2_3, xd2_4), dim=1)
        xd2 = self.ReLU(self.bnb2(xd2_con))

        xd1_1 = self.Dmp1_1_left(self.Dconv1_left(xd2))
        xd1_2 = self.Dmp1_3(self.Dconv1_3_3(self.Dconv1_3_1(xd2)))
        xd1_3 = self.Dmp1_5(self.Dconv1_5_5(self.Dconv1_5_1(xd2)))
        xd1_4 = self.Dconv1_right(self.Dmp1_1_right(xd2))
        xd1_con = torch.cat((xd1_1, xd1_2, xd1_3, xd1_4), dim=1)
        out_decoder = self.ReLU(self.bnb1(xd1_con))

        score = self.classifier(out_decoder)
        
        return score  # size=(N, n_class, x.H/1, x.W/1)