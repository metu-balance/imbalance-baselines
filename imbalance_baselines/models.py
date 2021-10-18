import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet32(nn.Module):
    def __init__(self, num_layers=32, num_classes=10):
        super(ResNet32, self).__init__()
        
        self.n = (num_layers - 2) // 6
        self.num_classes = num_classes
        self.filters = [16, 16, 32, 64]
        self.strides = [1, 2, 2]
        
        self.conv = nn.Conv2d(3, 16, 3, 1, padding='same', bias= False)
        self.norm = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        
        Layers = []
        
        for i in range(3):
            for j in range(self.n):
                if j == 0:
                    in_filter = self.filters[i]
                    stride = self.strides[i]
                
                else:
                    in_filter = self.filters[i+1]
                    stride = 1
                
                out_filter = self.filters[i+1]
                
                Layers.append(ResBlock(in_filter, out_filter, stride))
        
        self.sequential = nn.Sequential(*Layers)
        
        def global_avg_pool(x):
            return x.mean(axis= [2,3])
        
        self.global_pool = global_avg_pool
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.sequential(x)
        x = self.global_pool(x)
        x = self.fc(x)
        
        return x


class ResBlock(nn.Module):
    def __init__(self, in_filter, out_filter, stride):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_filter, out_filter, 3, stride, padding=1, bias= False)
        self.norm1 = nn.BatchNorm2d(out_filter)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_filter, out_filter, 3, 1, padding='same', bias= False)
        self.norm2 = nn.BatchNorm2d(out_filter)
        
        self.avg_pool = None
        
        def get_Padding(padding):
            pad = padding
            def Padding(x):
                return F.pad(x, padding)
            
            return Padding
        
        if (in_filter != out_filter):
            self.avg_pool = nn.AvgPool2d(stride, stride)
            self.pool_padding = get_Padding([0,1,0,1])
            pad = (out_filter - in_filter) // 2
            self.channel_padding = get_Padding([0,0,0,0,pad,pad])
    
    def forward(self, x):
        with torch.no_grad():
            x_orig = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        if self.avg_pool is not None:
            x_orig = self.pool_padding(x_orig)
            x_orig = self.avg_pool(x_orig)
            x_orig = self.channel_padding(x_orig)
        
        x += x_orig
        
        x = self.relu(x)
        
        return x

