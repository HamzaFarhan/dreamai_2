import sr_model
from utils import *
from model import *
import sr_unet_loss
import sr_model_loss
from dai_imports import*
from rdn import RDN, RDN_DN

class SuperRes(Network):
    def __init__(self,
                 model_name = 'sr_model',
                 model_type = 'super_res',
                 kornia_transforms = None,
                 lr = 0.02,
                 criterion = nn.L1Loss(),
                 perceptual_criterion = nn.L1Loss(),
                 loss_func = 'perecptual',
                 optimizer_name = 'sgd',
                 upscale_factor = 4,
                 growth_rate = 64,
                 rdb_number = 5,
                 rdb_conv_layers = 5,
                 res_blocks = 16,
                 device = None,
                 best_validation_loss = None,
                 best_model_file = 'best_super_res_sgd.pth'
                 ):

        super().__init__(device=device)
        model_modules = []
        if kornia_transforms is not None:
            self.kornia_transforms = nn.Sequential(*kornia_transforms)
            model_modules.append(self.kornia_transforms)
        else:
            self.kornia_transforms = None
        self.set_backbone(model_name=model_name,upscale_factor=upscale_factor,growth_rate=growth_rate,
                       rdb_number=rdb_number,rdb_conv_layers=rdb_conv_layers,res_blocks=res_blocks,device=device)
        model_modules.insert(0,self.backbone)
        self.model = nn.Sequential(*model_modules).to(device)
        self.set_model_params(criterion = criterion,optimizer_name = optimizer_name,lr = lr,model_name = model_name,model_type = model_type,
                              best_validation_loss = best_validation_loss,best_model_file = best_model_file)
        if loss_func.lower() == 'perceptual':
            self.feature_loss = sr_model_loss.FeatureLoss(2,[0.26,0.74],device = device)
        self.loss_func = loss_func
        

    def set_backbone(self,model_name,upscale_factor,growth_rate=64,
                  rdb_number=16,rdb_conv_layers=8,res_blocks=10,device='cpu'):
        print('Setting up Super Resolution model: ',end='')
        if model_name.lower() == 'rdn':
            print('Using RDN for super res.')
            self.backbone = RDN(channel=3,growth_rate=growth_rate,rdb_number=rdb_number,
                            rdb_conv_layers=rdb_conv_layers,upscale_factor=upscale_factor).to(device)
        elif model_name.lower() == 'sr_model':
            print('Using SrResnet for super res.')
            self.backbone = sr_model.SrResnet(scale=upscale_factor,res_blocks=res_blocks).to(device)
            
    def forward(self,x):
        if self.kornia_transforms is not None:
            for p in self.model[1].parameters():
                p.data = nn.Parameter(tensor(max(0.,p.data))).to(self.device)
        return self.model(x)

    def compute_loss(self,outputs,labels):

        ret = {}
        ret['mse'] = F.mse_loss(outputs,labels)
        if self.loss_func.lower() == 'crit':
            basic_loss = self.criterion(outputs, labels)
            ret['overall_loss'] = basic_loss
            return basic_loss,ret
        elif self.loss_func.lower() == 'perceptual':
            overall_loss = self.feature_loss(outputs,labels)
            ret['overall_loss'] = overall_loss
            return overall_loss,ret
    
    def evaluate(self,dataloader, **kwargs):
        
        running_loss = 0.
        running_psnr = 0.
        rmse_ = 0.
        self.eval()
        with torch.no_grad():
            for data_batch in dataloader:
                img, hr_target, hr_resized = data_batch[0],data_batch[1],data_batch[2]
                img = img.to(self.device)
                hr_target = hr_target.to(self.device)
                hr_super_res = self.forward(img)
                _,loss_dict = self.compute_loss(hr_super_res,hr_target)
                torchvision.utils.save_image([hr_target.cpu()[0],hr_resized[0],hr_super_res.cpu()[0]],filename='current_sr_model_performance.png')
                running_psnr += 10 * math.log10(1 / loss_dict['mse'].item())
                running_loss += loss_dict['overall_loss'].item()
                rmse_ += rmse(hr_super_res,hr_target).cpu().numpy()
        self.train()
        ret = {}
        ret['final_loss'] = running_loss/len(dataloader)
        ret['psnr'] = running_psnr/len(dataloader)
        ret['final_rmse'] = rmse_/len(dataloader)
        return ret

class SuperResUnet(Network):
    def __init__(self,
                 model_name = 'resnet34',
                 model_type = 'super_res',
                 kornia_transforms = None,
                 lr = 0.08,
                 upscale_factor = 4, 
                 criterion = sr_unet_loss.FeatureLoss(),
                 img_mean = [0.485, 0.456, 0.406],
                 img_std = [0.229, 0.224, 0.225],
                #  p_blocks_start = 2,
                #  p_blocks_end = 5,
                #  p_layer_wgts = [5,15,2],
                #  p_criterion = F.l1_loss,
                #  loss_func = 'perecptual',
                 optimizer_name = 'sgd',
                 device = None,
                 best_validation_loss = None,
                 best_psnr = None,
                 best_model_file = 'best_super_res_sgd.pth',
                 model_weights = None,
                 optim_weights = None
                 ):

        super().__init__(device=device)
        self.set_scale(upscale_factor)
        self.model = smp.Unet(encoder_name = model_name, classes = 3)
        if model_weights:
            self.model.load_state_dict(model_weights)
        self.model.to(device)
        # if loss_func.lower() == 'perceptual':
        #     criterion = sr_unet_loss.FeatureLoss(p_blocks_start,p_blocks_end,p_layer_wgts,p_criterion,device = device)
        self.set_model_params(criterion = criterion,optimizer_name = optimizer_name,lr = lr,model_name = model_name,model_type = model_type,
                              best_validation_loss = best_validation_loss,best_model_file = best_model_file)
        if optim_weights:
            self.optim.load_state_dict(optim_weights)
        self.best_psnr = best_psnr
        self.img_mean = img_mean
        self.img_std = img_std
        
    def set_scale(self,scale):
        self.upscale_factor = scale

    def forward(self,x):
        x = F.interpolate(x,scale_factor=self.upscale_factor,mode='bicubic')
        x = self.model(x)

        x[:, 0, :, :] = x[:, 0, :, :] * self.img_std[0] + self.img_mean[0]
        x[:, 1, :, :] = x[:, 1, :, :] * self.img_std[1] + self.img_mean[1]
        x[:, 2, :, :] = x[:, 2, :, :] * self.img_std[2] + self.img_mean[2]
        
        return x

    def compute_loss(self,outputs,labels):

        ret = {}
        ret['mse'] = F.mse_loss(outputs,labels)
        loss = self.criterion(outputs, labels)
        ret['overall_loss'] = loss
        return loss,ret
    
    def evaluate(self,dataloader, **kwargs):
        
        running_loss = 0.
        running_psnr = 0.
        rmse_ = 0.
        self.eval()
        with torch.no_grad():
            for data_batch in dataloader:
                img, hr_target, hr_resized = data_batch[0],data_batch[1],data_batch[2]
                img = img.to(self.device)
                hr_target = hr_target.to(self.device)
                hr_super_res = self.forward(img)
                _,loss_dict = self.compute_loss(hr_super_res,hr_target)
                torchvision.utils.save_image([hr_target.cpu()[0],hr_resized[0],hr_super_res.cpu()[0]],filename='current_sr_model_performance.png')
                running_psnr += 10 * math.log10(1 / loss_dict['mse'].item())
                running_loss += loss_dict['overall_loss'].item()
                rmse_ += rmse(hr_super_res,hr_target).cpu().numpy()
        self.train()
        ret = {}
        ret['final_loss'] = running_loss/len(dataloader)
        ret['psnr'] = running_psnr/len(dataloader)
        ret['final_rmse'] = rmse_/len(dataloader)
        return ret