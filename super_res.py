import dbpn_v1
import sr_resnet
import sr_densenet
from utils import *
from model import *
import feature_loss
import sr_model_loss
from dai_imports import*
from rdn import RDN, RDN_DN
from pixel_shuffle import PixelShuffle_ICNR
from dbpn_discriminator import Discriminator, FeatureExtractor, FeatureExtractorResnet

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
            self.backbone = sr_resnet.SrResnet(scale=upscale_factor,res_blocks=res_blocks).to(device)
            
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
                 criterion = feature_loss.FeatureLoss(),
                 img_mean = [0.485, 0.456, 0.406],
                 img_std = [0.229, 0.224, 0.225],
                 inter_mode = 'bicubic',
                 attention_type =  None,
                 shuffle_blur = True,
                 use_bn = True,
                 denorm = True,
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
                 encoder_weights = 'imagenet',
                 model_weights = None,
                 optim_weights = None
                 ):

        super().__init__(device=device)

        print(f'Super Resolution using U-Net with {model_name} encoder.')

        self.set_scale(upscale_factor)
        self.set_inter_mode(inter_mode)
        self.set_denorm(denorm)
        self.setup_model(model_name, encoder_weights,  use_bn, attention_type, shuffle_blur)
        if model_weights:
            self.model.load_state_dict(model_weights)
        self.model.to(device)
        self.set_model_params(criterion = criterion,optimizer_name = optimizer_name,lr = lr,model_name = model_name,model_type = model_type,
                              best_validation_loss = best_validation_loss,best_model_file = best_model_file)
        if optim_weights:
            self.optim.load_state_dict(optim_weights)
        self.best_psnr = best_psnr
        self.img_mean = img_mean
        self.img_std = img_std

    def setup_model(self, model_name='resnext50_32x4d', encoder_weights='imagenet', use_bn=False, attention_type=None, shuffle_blur=True):
        unet = smp.Unet(encoder_name=model_name, encoder_weights=encoder_weights, classes=3,
                        decoder_use_batchnorm=use_bn, attention_type=attention_type, shuffle_blur=shuffle_blur)
        # if self.inter_mode.lower() == 'shuffle':
        #     num_shuffles = self.upscale_factor//2
        #     shuffle = [PixelShuffle_ICNR(3, 3, scale=2, blur=True)]*num_shuffles
        #     # shuffle = [PixelShuffle_ICNR(3, 3, scale=self.upscale_factor, blur=True)]
        #     self.model = nn.Sequential(*shuffle,unet)
        #     # conv1 = nn.Conv2d(3,self.upscale_factor**2,3,1,1)
        #     # conv2 = nn.Conv2d(1,3,3,1,1)
        #     # self.model = nn.Sequential(conv1, nn.PixelShuffle(self.upscale_factor), conv2, unet)
        if self.inter_mode.lower() == 'shuffle':
            shuffle = PixelShuffle_ICNR(3, 3, self.upscale_factor, shuffle_blur)
            unet_ext = [conv_block(3,3,3,1,1,True,False), shuffle,
                        nn.BatchNorm2d(3),
                        conv_block(3,3,3,1,1,False,False)]
            self.model = nn.Sequential(unet,*unet_ext)
        else:
            self.model = nn.Sequential(unet)

    def set_denorm(self,denorm=True):
        self.denorm = denorm

    def set_scale(self,scale):
        self.upscale_factor = scale

    def set_inter_mode(self,mode):
        self.inter_mode = mode

    def forward(self,x):
        if self.inter_mode.lower() != 'shuffle':
            x = F.interpolate(x,scale_factor=self.upscale_factor,mode=self.inter_mode)
        x = self.model(x)
        if self.denorm:
            x = denorm_tensor(x, self.img_mean, self.img_std)
            # x[:, 0, :, :] = x[:, 0, :, :] * self.img_std[0] + self.img_mean[0]
            # x[:, 1, :, :] = x[:, 1, :, :] * self.img_std[1] + self.img_mean[1]
            # x[:, 2, :, :] = x[:, 2, :, :] * self.img_std[2] + self.img_mean[2]
        
        return x

    def freeze_encoder(self):
        for p in self.model[0].encoder.parameters():
            p.requires_grad = False

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
                torchvision.utils.save_image([
                                            #   denorm_tensor(hr_target.cpu()[0], self.img_mean, self.img_std),
                                              hr_target.cpu()[0],
                                              hr_resized[0],
                                              hr_super_res.cpu()[0]
                                              ],
                                              filename='current_sr_model_performance.png')
                running_psnr += 10 * math.log10(1 / loss_dict['mse'].item())
                running_loss += loss_dict['overall_loss'].item()
                rmse_ += rmse(hr_super_res,hr_target).cpu().numpy()
        self.train()
        ret = {}
        ret['final_loss'] = running_loss/len(dataloader)
        ret['psnr'] = running_psnr/len(dataloader)
        ret['final_rmse'] = rmse_/len(dataloader)
        return ret

class SrNetwork(Network):
    def __init__(self,
                 model_name = 'resnet',
                 model_type = 'super_res',
                 kornia_transforms = None,
                 lr = 0.08,
                 upscale_factor = 4,
                 num_blocks = 8,
                 block_channels = 32,
                 block_scale = 0.1,
                 shuffle_blur = True,
                 criterion = feature_loss.FeatureLoss(),
                 optimizer_name = 'sgd',
                 img_mean = [0.485, 0.456, 0.406],
                 img_std = [0.229, 0.224, 0.225],
                 denorm = True,
                 device = None,
                 best_validation_loss = None,
                 best_psnr = None,
                 best_model_file = 'best_sr_resnet_sgd.pth',
                 ):

        super().__init__(device=device)

        self.set_scale(upscale_factor)
        self.set_model(model_name=model_name, scale=upscale_factor, num_blocks=num_blocks,
                       block_channels=block_channels, block_scale=block_scale, shuffle_blur=shuffle_blur)
        self.model.to(device)
        self.set_model_params(criterion=criterion, optimizer_name=optimizer_name, lr=lr, model_name=model_name, model_type=model_type,
                              best_validation_loss=best_validation_loss, best_model_file=best_model_file)
        self.best_psnr = best_psnr
        self.img_mean = img_mean
        self.img_std = img_std
        self.denorm = denorm

    def set_scale(self,scale):
        self.upscale_factor = scale

    def set_model(self,model_name='resnet', scale=4, num_blocks=8, block_channels=64, block_scale=0.1, shuffle_blur=True):
        model_dict = {'resnet': sr_resnet.SrResnet, 'densenet': sr_densenet.SrDensenet}
        assert model_name.lower() in model_dict, print(f'Please use one of "densenet" or "resnet" as your model.')
        args = [scale, num_blocks, block_channels, block_scale, shuffle_blur]
        self.model = model_dict[model_name.lower()](*args)
        print(f'Super Resolution using {model_name}.')

    def forward(self,x):
        x = self.model(x)
        if self.denorm:
            x = denorm_tensor(x, self.img_mean, self.img_std)
            # x[:, 0, :, :] = x[:, 0, :, :] * self.img_std[0] + self.img_mean[0]
            # x[:, 1, :, :] = x[:, 1, :, :] * self.img_std[1] + self.img_mean[1]
            # x[:, 2, :, :] = x[:, 2, :, :] * self.img_std[2] + self.img_mean[2]
        return x

    def freeze(self,idx):
        for i in idx:
            for p in self.model.features[i].parameters():
                p.requires_grad = False

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
                torchvision.utils.save_image([
                                            #   denorm_tensor(hr_target.cpu()[0], self.img_mean, self.img_std),
                                              hr_target.cpu()[0],
                                              hr_resized[0],
                                              hr_super_res.cpu()[0]
                                              ],
                                              filename='current_sr_model_performance.png')
                running_psnr += 10 * math.log10(1 / loss_dict['mse'].item())
                running_loss += loss_dict['overall_loss'].item()
                rmse_ += rmse(hr_super_res,hr_target).cpu().numpy()
        self.train()
        ret = {}
        ret['final_loss'] = running_loss/len(dataloader)
        ret['psnr'] = running_psnr/len(dataloader)
        ret['final_rmse'] = rmse_/len(dataloader)
        return ret

class SuperResDBPN(Network):
    def __init__(self,
                 model_name = 'dbpn',
                 model_type = 'super_res',
                 lr = 0.08,
                 num_channels = 3,
                 base_filter = 64,
                 feat = 256,
                 upscale_factor = 4, 
                 criterion = nn.L1Loss(),
                 img_mean = [0.485, 0.456, 0.406],
                 img_std = [0.229, 0.224, 0.225],
                 inter_mode = 'bicubic',
                 residual = False,
                 denorm = False,
                 optimizer_name = 'adam',
                 device = None,
                 best_validation_loss = None,
                 best_psnr = None,
                 best_model_file = 'best_dbpn_sgd.pth',
                 model_weights = None,
                 optim_weights = None
                 ):

        super().__init__(device=device)

        print(f'Super Resolution using DBPN.')

        self.set_inter_mode(inter_mode)
        self.set_scale(upscale_factor)
        self.set_residual(residual)
        self.set_denorm(denorm)
        self.model = dbpn_v1.Net(num_channels=num_channels, base_filter=base_filter,
                                 feat=feat, num_stages=10, scale_factor=upscale_factor)
        # print(self.model.state_dict().keys())
        if model_weights:
            self.model.load_state_dict(model_weights)
        modules = list(self.model.module.named_modules())
        for n,p in modules:
            if isinstance(p, nn.Conv2d):
                setattr(self.model.module, n, nn.utils.weight_norm(p))
        self.model.to(device)
        self.set_model_params(criterion = criterion,optimizer_name = optimizer_name,lr = lr,model_name = model_name,model_type = model_type,
                              best_validation_loss = best_validation_loss,best_model_file = best_model_file)
        if optim_weights:
            self.optim.load_state_dict(optim_weights)
        self.best_psnr = best_psnr
        self.img_mean = img_mean
        self.img_std = img_std

    def set_denorm(self,denorm=True):
        self.denorm = denorm

    def set_scale(self,scale):
        self.upscale_factor = scale

    def set_inter_mode(self,mode):
        self.inter_mode = mode

    def set_residual(self,res):
        self.residual = res

    def forward(self,x):
        if self.inter_mode is not None:
            res = F.interpolate(x.clone().detach(), scale_factor=self.upscale_factor, mode=self.inter_mode)
        x = self.model(x)
        if self.residual:
            x += res
        if self.denorm:
            x = denorm_tensor(x, self.img_mean, self.img_std)
            # x[:, 0, :, :] = x[:, 0, :, :] * self.img_std[0] + self.img_mean[0]
            # x[:, 1, :, :] = x[:, 1, :, :] * self.img_std[1] + self.img_mean[1]
            # x[:, 2, :, :] = x[:, 2, :, :] * self.img_std[2] + self.img_mean[2]
        
        return x

    def compute_loss(self,outputs,labels):

        ret = {}
        ret['mse'] = F.mse_loss(outputs,labels)
        loss = self.criterion(outputs, labels)
        ret['overall_loss'] = loss
        return loss,ret
    
    def evaluate(self,dataloader, **kwargs):

        # res = self.residual
        # self.set_residual(False)
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
                torchvision.utils.save_image([
                                            #   denorm_tensor(hr_target.cpu()[0], self.img_mean, self.img_std),
                                              hr_target.cpu()[0],
                                              hr_resized[0],
                                              hr_super_res.cpu()[0]
                                              ],
                                              filename='current_sr_model_performance.png')
                running_psnr += 10 * math.log10(1 / loss_dict['mse'].item())
                running_loss += loss_dict['overall_loss'].item()
                rmse_ += rmse(hr_super_res,hr_target).cpu().numpy()
        # self.set_residual(res)
        self.train()
        ret = {}
        ret['final_loss'] = running_loss/len(dataloader)
        ret['psnr'] = running_psnr/len(dataloader)
        ret['final_rmse'] = rmse_/len(dataloader)
        return ret

    # def predict(self,inputs,actv = None):
    #     res = self.residual
    #     self.set_residual(False)
    #     self.eval()
    #     self.model.eval()
    #     self.model = self.model.to(self.device)
    #     with torch.no_grad():
    #         inputs = inputs.to(self.device)
    #         outputs = self.forward(inputs)
    #     if actv is not None:
    #         return actv(outputs)
    #     self.set_residual(res)
    #     return outputs

class SuperResDBPNGAN(Network):
    def __init__(self,
                 model_name = 'dbpn_gan',
                 model_type = 'super_res',
                 lr = 0.001,
                 d_lr = 0.001,
                 num_channels = 3,
                 base_filter = 64,
                 feat = 256,
                 upscale_factor = 4, 
                #  criterion = nn.MSELoss(),
                 criterion = feature_loss.FeatureLoss(),
                 adv_loss_weight = 1e-3,
                 img_mean = [0.485, 0.456, 0.406],
                 img_std = [0.229, 0.224, 0.225],
                 denorm = False,
                 normed_data = False,
                 optimizer_name = 'adam',
                 d_optimizer_name = 'adam',
                 device = None,
                 best_validation_loss = None,
                 best_psnr = None,
                 best_model_file = 'best_dbpngan_adam.pth',
                 gen_weights = None,
                 disc_weights = None,
                 optim_weights = None,
                 sr_image_size = 224
                 ):

        super().__init__(device=device)

        print(f'Super Resolution using DBPN GAN.')

        self.adv_loss_weight = adv_loss_weight
        self.set_scale(upscale_factor)
        self.set_denorm(denorm)
        self.G = dbpn_v1.Net(num_channels=num_channels, base_filter=base_filter,
                                 feat=feat, num_stages=10, scale_factor=upscale_factor)
        self.D = Discriminator(num_channels=num_channels, base_filter=base_filter, image_size=sr_image_size)
        # self.feature_loss = feature_loss
        # self.feature_loss.set_base_loss(criterion)
        # self.feature_extractor = FeatureExtractor(models.vgg16(pretrained=True)).to(device).eval()
        # for p in self.feature_extractor.parameters():
            # p.requires_grad = False
        # print(self.model.state_dict().keys())
        if gen_weights:
            self.G.load_state_dict(gen_weights)
        if disc_weights:
            self.D.load_state_dict(disc_weights)
        self.model = nn.ModuleDict({'G':self.G, 'D':self.D}).to(device)
        self.set_model_params(params=self.G.parameters() ,optimizer_name=optimizer_name, lr=lr,
                              d_params=self.D.parameters() ,d_optimizer_name=d_optimizer_name, d_lr=d_lr,
                              criterion=criterion, model_name=model_name, model_type=model_type,
                              best_validation_loss=best_validation_loss, best_model_file=best_model_file)
        if optim_weights:
            self.optim.load_state_dict(optim_weights)
        self.best_psnr = best_psnr
        self.img_mean = img_mean
        self.img_std = img_std

    def set_denorm(self,denorm=True):
        self.denorm = denorm

    def set_scale(self,scale):
        self.upscale_factor = scale

    def forward(self,x):
        x = self.model.G(x)
        if self.denorm:
            x = denorm_tensor(x, self.img_mean, self.img_std)
            # x[:, 0, :, :] = x[:, 0, :, :] * self.img_std[0] + self.img_mean[0]
            # x[:, 1, :, :] = x[:, 1, :, :] * self.img_std[1] + self.img_mean[1]
            # x[:, 2, :, :] = x[:, 2, :, :] * self.img_std[2] + self.img_mean[2]
        
        return x

    def compute_loss(self,outputs,labels):

        ret = {}
        ret['mse'] = F.mse_loss(outputs,labels)
        loss = self.criterion(outputs, labels)
        ret['overall_loss'] = loss
        return loss,ret
    
    def train_(self, e, trainloader, optimizer, print_every, clip=False):
        
        BCE_loss = nn.BCELoss()
        G_epoch_loss = 0.
        D_epoch_loss = 0.
        feat_epoch_loss = 0.
        style_epoch_loss = 0.
        adv_epoch_loss = 0.
        crit_epoch_loss = 0.
        gen_epoch_loss = 0.

        self.train()
        epoch,epochs = e
        t0 = time.time()
        t1 = time.time()
        batches = 0
        for data_batch in trainloader:
            batches += 1
            inputs,target = data_batch[0],data_batch[1]
            minibatch = inputs.shape[0]
            real_label = torch.ones(minibatch).to(self.device) #torch.rand(minibatch,1)*0.5 + 0.7
            fake_label = torch.zeros(minibatch).to(self.device) #torch.rand(minibatch,1)*0.3
            inputs = inputs.to(self.device)
            target = target.to(self.device)
            # outputs = self.forward(inputs)
            # loss = self.compute_loss(outputs,labels)[0]

            # Reset gradient
            self.d_optimizer.zero_grad()
            
            # Train discriminator with real data
            D_real_decision = self.model.D(target).squeeze(1)
            D_real_loss = BCE_loss(D_real_decision, real_label)
            
            # Train discriminator with fake data
            recon_image = self.model.G(inputs)
            D_fake_decision = self.model.D(recon_image).squeeze(1)
            D_fake_loss = BCE_loss(D_fake_decision, fake_label)
            
            D_loss = D_real_loss + D_fake_loss
            
            # Back propagation
            D_loss.backward()
            self.d_optimizer.step()
            
            # Reset gradient
            optimizer.zero_grad()
            
            # Train generator
            recon_image = self.model.G(inputs)
            D_fake_decision = self.model.D(recon_image).squeeze(1)
            
            # Adversarial loss
            GAN_loss = self.adv_loss_weight * BCE_loss(D_fake_decision, real_label)
            
            gen_loss = self.criterion(recon_image, target)

            # Content losses
            # crit_loss = self.loss_weights[0] * self.criterion(recon_image, target)
            
            # #Perceptual loss
            # x_VGG = target.data.clone()
            # recon_VGG = recon_image.data.clone()
            # real_feature = self.feature_extractor(x_VGG)
            # fake_feature = self.feature_extractor(recon_VGG)
            # # torch.autograd.set_detect_anomaly(True)
            # vgg_loss = self.loss_weights[1] * sum([self.criterion(fake_feature[i], real_feature[i].detach()) for i in range(len(real_feature))])        
            # style_loss = self.loss_weights[3] * sum([self.criterion(gram_matrix(fake_feature[i]),
            #                                          gram_matrix(real_feature[i]).detach()) for i in range(len(real_feature))])

            # # Back propagation
            # gen_loss = crit_loss + vgg_loss + style_loss

            G_loss = gen_loss + GAN_loss

            G_loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(self.model.parameters(),1.)
            optimizer.step()

            G_epoch_loss += G_loss.item()
            D_epoch_loss += D_loss.item()
            # feat_epoch_loss += (vgg_loss.item())
            # style_epoch_loss += (style_loss.item())
            adv_epoch_loss += (GAN_loss.item())
            # crit_epoch_loss += (crit_loss.item())
            gen_epoch_loss += (gen_loss.item())

            if batches % print_every == 0:
                elapsed = time.time()-t1
                if elapsed > 60:
                    elapsed /= 60.
                    measure = 'min'
                else:
                    measure = 'sec'
                batch_time = time.time()-t0
                if batch_time > 60:
                    batch_time /= 60.
                    measure2 = 'min'
                else:
                    measure2 = 'sec'    
                print('+----------------------------------------------------------------------+\n'
                        f"{time.asctime().split()[-2]}\n"
                        f"Time elapsed: {elapsed:.3f} {measure}\n"
                        f"Epoch:{epoch+1}/{epochs}\n"
                        f"Batch: {batches+1}/{len(trainloader)}\n"
                        f"Batch training time: {batch_time:.3f} {measure2}\n"
                        f"Batch training loss: {G_loss.item():.3f}\n"
                        f"Average Descriminator loss: {D_epoch_loss/(batches):.3f}\n"
                        f"Average Generator loss: {G_epoch_loss/(batches):.3f}\n"
                        # f"Average feature loss: {feat_epoch_loss/(batches):.3f}\n"
                        # f"Average style loss: {feat_epoch_loss/(batches):.3f}\n"
                        f"Average adversarial loss: {adv_epoch_loss/(batches):.3f}\n"
                        f"Average generator criterion loss: {gen_epoch_loss/(batches):.3f}\n"
                        # f"Average criterion loss: {crit_epoch_loss/(batches):.3f}\n"
                      '+----------------------------------------------------------------------+\n'     
                        )
                t0 = time.time()
        return G_epoch_loss/len(trainloader)

    def set_optimizer(self, params, d_params, optimizer_name='adam', lr=0.003,
                      d_optimizer_name='adam', d_lr=0.003 ):
        if optimizer_name:
            optimizer_name = optimizer_name.lower()
            if optimizer_name == 'adam':
                print('Setting generator optimizer: Adam')
                self.optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
                self.optimizer_name = optimizer_name
            elif optimizer_name == 'sgd':
                print('Setting generator optimizer: SGD')
                self.optimizer = optim.SGD(params,lr=lr)
                self.optimizer_name = optimizer_name
            elif optimizer_name == 'adadelta':
                print('Setting generator optimizer: AdaDelta')
                # self.optimizer = optim.Adadelta(params)
                self.optimizer = optim.Adadelta(params,lr=lr)
                self.optimizer_name = optimizer_name

            d_optimizer_name = d_optimizer_name.lower()
            if d_optimizer_name == 'adam':
                print('Setting discriminator optimizer: Adam')
                self.d_optimizer = optim.Adam(d_params, lr=d_lr, betas=(0.9, 0.999), eps=1e-8)
                self.d_optimizer_name = d_optimizer_name
            elif d_optimizer_name == 'sgd':
                print('Setting discriminator optimizer: SGD')
                self.d_optimizer = optim.SGD(d_params,lr=d_lr)
                self.d_optimizer_name = d_optimizer_name
            elif d_optimizer_name == 'adadelta':
                print('Setting discriminator optimizer: AdaDelta')
                # self.optimizer = optim.Adadelta(params)
                self.d_optimizer = optim.Adadelta(d_params,lr=d_lr)
                self.d_optimizer_name = d_optimizer_name
            
    def set_model_params(self,
                         params = None,
                         optimizer_name = 'sgd',
                         lr = 0.01,
                         d_params = None,
                         d_optimizer_name = 'sgd',
                         d_lr = 0.01,
                         criterion = nn.CrossEntropyLoss(),
                         dropout_p = 0.45,
                         model_name = 'resnet50',
                         model_type = 'classifier',
                         best_accuracy = 0.,
                         best_validation_loss = None,
                         best_model_file = 'best_model_file.pth'):        
        if params is None:
            params = self.G.parameters()
        if d_params is None:
            d_params = self.D.parameters()
        self.set_criterion(criterion)
        self.optimizer_name = optimizer_name
        self.set_optimizer(params=params, optimizer_name=optimizer_name, lr=lr,
                           d_params=d_params, d_optimizer_name=d_optimizer_name, d_lr=d_lr)
        self.lr = lr
        self.dropout_p = dropout_p
        self.model_name =  model_name
        self.model_type = model_type
        self.best_accuracy = best_accuracy
        self.best_validation_loss = best_validation_loss
        self.best_model_file = best_model_file

    def evaluate(self,dataloader, **kwargs):

        # res = self.residual
        # self.set_residual(False)
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
                # print(hr_target.shape,hr_super_res.shape, hr_resized.shape)
                _,loss_dict = self.compute_loss(hr_super_res,hr_target)
                torchvision.utils.save_image([
                                            #   denorm_tensor(hr_target.cpu()[0], self.img_mean, self.img_std).squeeze(0),
                                              hr_target.cpu()[0],
                                              hr_resized[0],
                                            #   denorm_tensor(hr_super_res.cpu()[0], self.img_mean, self.img_std).squeeze(0),
                                              hr_super_res.cpu()[0]
                                              ],
                                              filename='current_sr_model_performance.png')
                running_psnr += 10 * math.log10(1 / loss_dict['mse'].item())
                running_loss += loss_dict['overall_loss'].item()
                rmse_ += rmse(hr_super_res,hr_target).cpu().numpy()
        # self.set_residual(res)
        self.train()
        ret = {}
        ret['final_loss'] = running_loss/len(dataloader)
        ret['psnr'] = running_psnr/len(dataloader)
        ret['final_rmse'] = rmse_/len(dataloader)
        return ret

    # def predict(self,inputs,actv = None):
    #     res = self.residual
    #     self.set_residual(False)
    #     self.eval()
    #     self.model.eval()
    #     self.model = self.model.to(self.device)
    #     with torch.no_grad():
    #         inputs = inputs.to(self.device)
    #         outputs = self.forward(inputs)
    #     if actv is not None:
    #         return actv(outputs)
    #     self.set_residual(res)
    #     return outputs