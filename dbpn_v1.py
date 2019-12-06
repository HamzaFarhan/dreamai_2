import os
import torch.nn as nn
import torch.optim as optim
from dbpn_base_networks import *
from torchvision.transforms import *

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, scale_factor):
        super(Net, self).__init__()
        
        if scale_factor == 2:
        	kernel = 6
        	stride = 2
        	padding = 2
        elif scale_factor == 4:
        	kernel = 8
        	stride = 4
        	padding = 2
        elif scale_factor == 8:
        	kernel = 12
        	stride = 8
        	padding = 2
        module_dict = {}
        #Initial Feature Extraction
        module_dict['feat0'] = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        module_dict['feat1'] = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        #Back-projection stages
        module_dict['up1'] = UpBlock(base_filter, kernel, stride, padding)
        module_dict['down1'] = DownBlock(base_filter, kernel, stride, padding)
        module_dict['up2'] = UpBlock(base_filter, kernel, stride, padding)
        module_dict['down2'] = D_DownBlock(base_filter, kernel, stride, padding, 2)
        module_dict['up3'] = D_UpBlock(base_filter, kernel, stride, padding, 2)
        module_dict['down3'] = D_DownBlock(base_filter, kernel, stride, padding, 3)
        module_dict['up4'] = D_UpBlock(base_filter, kernel, stride, padding, 3)
        module_dict['down4'] = D_DownBlock(base_filter, kernel, stride, padding, 4)
        module_dict['up5'] = D_UpBlock(base_filter, kernel, stride, padding, 4)
        module_dict['down5'] = D_DownBlock(base_filter, kernel, stride, padding, 5)
        module_dict['up6'] = D_UpBlock(base_filter, kernel, stride, padding, 5)
        module_dict['down6'] = D_DownBlock(base_filter, kernel, stride, padding, 6)
        module_dict['up7'] = D_UpBlock(base_filter, kernel, stride, padding, 6)
        module_dict['down7'] = D_DownBlock(base_filter, kernel, stride, padding, 7)
        module_dict['up8'] = D_UpBlock(base_filter, kernel, stride, padding, 7)
        module_dict['down8'] = D_DownBlock(base_filter, kernel, stride, padding, 8)
        module_dict['up9'] = D_UpBlock(base_filter, kernel, stride, padding, 8)
        module_dict['down9'] = D_DownBlock(base_filter, kernel, stride, padding, 9)
        module_dict['up10'] = D_UpBlock(base_filter, kernel, stride, padding, 9)
        #Reconstruction
        module_dict['output_conv'] = ConvBlock(num_stages*base_filter, num_channels, 3, 1, 1, activation=None, norm=None)
        self.module = nn.ModuleDict(module_dict)

        for m in self.module.values():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            
    def forward(self, x):
        x = self.module.feat0(x)
        x = self.module.feat1(x)
        
        h1 = self.module.up1(x)
        l1 = self.module.down1(h1)
        h2 = self.module.up2(l1)
        
        concat_h = torch.cat((h2, h1),1)
        l = self.module.down2(concat_h)
        
        concat_l = torch.cat((l, l1),1)
        h = self.module.up3(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.module.down3(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.module.up4(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.module.down4(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.module.up5(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.module.down5(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.module.up6(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.module.down6(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.module.up7(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.module.down7(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.module.up8(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.module.down8(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.module.up9(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.module.down9(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.module.up10(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        x = self.module.output_conv(concat_h)
        
        return x
