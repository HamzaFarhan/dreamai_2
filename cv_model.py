from dreamai import SSD
from dreamai import SSD300
from dreamai import obj_utils
from dreamai.fc import *
from dreamai.model import *
from dreamai.utils import *
from dreamai.dai_imports import*
from dreamai.lenet_plus import LeNetPlus
from dreamai.center_loss import CenterLoss
from dreamai.seg_hrnet import get_seg_model
from dreamai.efficientnet import EfficientNet
from dreamai.parallel import DataParallelModel, DataParallelCriterion

class FoodIngredients(Network):
    def __init__(self,
                 model_name='DenseNet',
                 model_type='food',
                 lr=0.02,
                 optimizer_name = 'Adam',
                 criterion1 = nn.CrossEntropyLoss(),
                 criterion2 = nn.BCEWithLogitsLoss(),
                 dropout_p=0.45,
                 pretrained=True,
                 device=None,
                 best_accuracy=0.,
                 best_validation_loss=None,
                 best_model_file ='best_model.pth',
                 head1 = {'num_outputs':10,
                    'layers':[],
                    'model_type':'classifier'
                 },
                 head2 = {'num_outputs':10,
                    'layers':[],
                    'model_type':'multi_label_classifier'
                 },
                 class_names = [],
                 num_classes = None,
                 ingredient_names = [],
                 num_ingredients = None,
                 add_extra = True,
                 set_params = True,
                 set_head = True
                 ):

        super().__init__(device=device)

        self.set_transfer_model(model_name,pretrained=pretrained,add_extra=add_extra,dropout_p=dropout_p)

        if set_head:
            self.set_model_head(model_name = model_name,
                    head1 = head1,
                    head2 = head2,
                    dropout_p = dropout_p,
                    criterion1 = criterion1,
                    criterion2 = criterion2,
                    device = device
                )
        if set_params:
            self.set_model_params(
                              optimizer_name = optimizer_name,
                              lr = lr,
                              dropout_p = dropout_p,
                              model_name = model_name,
                              model_type = model_type,
                              best_accuracy = best_accuracy,
                              best_validation_loss = best_validation_loss,
                              best_model_file = best_model_file,
                              class_names = class_names,
                              num_classes = num_classes,
                              ingredient_names = ingredient_names, 
                              num_ingredients = num_ingredients,
                              )

        self.model = self.model.to(device)
        
    def set_model_params(self,
                         criterion1 = nn.CrossEntropyLoss(),
                         criterion2 = nn.BCEWithLogitsLoss(),
                         optimizer_name = 'Adam',
                         lr  = 0.1,
                         dropout_p = 0.45,
                         model_name = 'DenseNet',
                         model_type = 'cv_transfer',
                         best_accuracy = 0.,
                         best_validation_loss = None,
                         best_model_file = 'best_model_file.pth',
                         head1 = {'num_outputs':10,
                                'layers':[],
                                'model_type':'classifier'
                                },
                         head2 = {'num_outputs':10,
                                'layers':[],
                                'model_type':'muilti_label_classifier'
                                },       
                         class_names = [],
                         num_classes = None,
                         ingredient_names = [],
                         num_ingredients = None):
        
        print('Food Names: current best accuracy = {:.3f}'.format(best_accuracy))
        if best_validation_loss is not None:
            print('Food Ingredients: current best loss = {:.3f}'.format(best_validation_loss))

        
        super(FoodIngredients, self).set_model_params(
                                              optimizer_name = optimizer_name,
                                              lr = lr,
                                              dropout_p = dropout_p,
                                              model_name = model_name,
                                              model_type = model_type,
                                              best_accuracy = best_accuracy,
                                              best_validation_loss = best_validation_loss,
                                              best_model_file = best_model_file
                                              )
        self.class_names = class_names
        self.num_classes = num_classes                                              
        self.ingredeint_names = ingredient_names
        self.num_ingredients = num_ingredients
        self.criterion1 = criterion1
        self.criterion2 = criterion2

    def forward(self,x):
        l = list(self.model.children())
        for m in l[:-2]:
            x = m(x)
        food = l[-2](x)
        ingredients = l[-1](x)
        return (food,ingredients)
    
    def compute_loss(self,outputs,labels,w1 = 1.,w2 = 1.): 
        out1,out2 = outputs
        label1,label2 = labels
        loss1 = self.criterion1(out1,label1)
        loss2 = self.criterion2(out2,label2)
        return [(loss1*w1)+(loss2*w2)]

    def freeze(self,train_classifier=True):
        super(FoodIngredients, self).freeze()
        if train_classifier:
            for param in self.model.fc1.parameters():
                 param.requires_grad = True
            for param in self.model.fc2.parameters():
                 param.requires_grad = True     

    def parallelize(self):
        self.parallel = True
        self.model = DataParallelModel(self.model)
        self.criterion  = DataParallelCriterion(self.criterion)

    def set_transfer_model(self,mname,pretrained=True,add_extra=True,dropout_p = 0.45):   
        self.model = None
        models_dict = {

            'densenet': {'model':models.densenet121(pretrained=pretrained),'conv_channels':1024},
            'resnet34': {'model':models.resnet34(pretrained=pretrained),'conv_channels':512},
            'resnet50': {'model':models.resnet50(pretrained=pretrained),'conv_channels':2048}

        }
        meta = models_dict[mname.lower()]
        try:
            model = meta['model']
            for param in model.parameters():
                param.requires_grad = False
            self.model = model    
            print('Setting transfer learning model: self.model set to {}'.format(mname))
        except:
            print('Setting transfer learning model: model name {} not supported'.format(mname))            

        # creating and adding extra layers to the model
        dream_model = None
        if add_extra:
            channels = meta['conv_channels']
            dream_model = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,1),
                # Printer(),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p),
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p),
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p)
                )        
        self.dream_model = dream_model          
           
    def set_model_head(self,
                        model_name = 'DenseNet',
                        head1 = {'num_outputs':10,
                                'layers':[],
                                'class_names': None,
                                'model_type':'classifier'
                               },
                        head2 = {'num_outputs':10,
                                'layers':[],
                                'class_names': None,
                                'model_type':'muilti_label_classifier'
                               },       
                        criterion1 = nn.CrossEntropyLoss(),
                        criterion2 = nn.BCEWithLogitsLoss(), 
                        adaptive = True,       
                        dropout_p = 0.45,
                        device = None):

        models_meta = {
        'resnet34': {'conv_channels':512,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'resnet50': {'conv_channels':2048,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'densenet': {'conv_channels':1024,'head_id': -1,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool]
                    ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1)]}
        }

        name = model_name.lower()
        meta = models_meta[name]
        modules = list(self.model.children())
        l = modules[:meta['head_id']]
        if self.dream_model:
            l+=self.dream_model
        heads = [head1,head2]
        crits = [criterion1,criterion2]    
        fcs = []
        for head,criterion in zip(heads,crits):
            head['criterion'] = criterion
            if head['model_type'].lower() == 'classifier':
                head['output_non_linearity'] = None
            fc = modules[-1]
            try:
                in_features =  fc.in_features
            except:
                in_features = fc.model.out.in_features    
            fc = FC(
                    num_inputs = in_features,
                    num_outputs = head['num_outputs'],
                    layers = head['layers'],
                    model_type = head['model_type'],
                    output_non_linearity = head['output_non_linearity'],
                    dropout_p = dropout_p,
                    criterion = head['criterion'],
                    optimizer_name = None,
                    device = device
                    )
            fcs.append(fc)          
        if adaptive:
            l += meta['adaptive_head']
        else:
            l += meta['normal_head']
        model = nn.Sequential(*l)
        model.add_module('fc1',fcs[0])
        model.add_module('fc2',fcs[1])
        self.model = model
        self.head1 = head1
        self.head2 = head2
        
        print('Multi-head set up complete.')

    def train_(self,e,trainloader,optimizer,print_every):

        epoch,epochs = e
        self.train()
        t0 = time.time()
        t1 = time.time()
        batches = 0
        running_loss = 0.
        for data_batch in trainloader:
            inputs,label1,label2 = data_batch[0],data_batch[1],data_batch[2]
            batches += 1
            inputs = inputs.to(self.device)
            label1 = label1.to(self.device)
            label2 = label2.to(self.device)
            labels = (label1,label2)
            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.compute_loss(outputs,labels)[0]
            if self.parallel:
                loss.sum().backward()
                loss = loss.sum()
            else:    
                loss.backward()
                loss = loss.item()
            optimizer.step()
            running_loss += loss
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
                        f"Batch training loss: {loss:.3f}\n"
                        f"Average training loss: {running_loss/(batches):.3f}\n"
                      '+----------------------------------------------------------------------+\n'     
                        )
                t0 = time.time()
        return running_loss/len(trainloader) 

    def evaluate(self,dataloader,metric='accuracy'):
        
        running_loss = 0.
        classifier = None

        if self.model_type == 'classifier':# or self.num_classes is not None:
           classifier = Classifier(self.class_names)

        y_pred = []
        y_true = []

        self.eval()
        rmse_ = 0.
        with torch.no_grad():
            for data_batch in dataloader:
                inputs,label1,label2 = data_batch[0],data_batch[1],data_batch[2]
                inputs = inputs.to(self.device)
                label1 = label1.to(self.device)
                label2 = label2.to(self.device)
                labels = (label1,label2)
                outputs = self.forward(inputs)
                loss = self.compute_loss(outputs,labels)[0]
                if self.parallel:
                    running_loss += loss.sum()
                    outputs = parallel.gather(outputs,self.device)
                else:        
                    running_loss += loss.item()
                if classifier is not None and metric == 'accuracy':
                    classifier.update_accuracies(outputs,labels)
                    y_true.extend(list(labels.squeeze(0).cpu().numpy()))
                    _, preds = torch.max(torch.exp(outputs), 1)
                    y_pred.extend(list(preds.cpu().numpy()))
                elif metric == 'rmse':
                    rmse_ += rmse(outputs,labels).cpu().numpy()
            
        self.train()

        ret = {}
        # print('Running_loss: {:.3f}'.format(running_loss))
        if metric == 'rmse':
            print('Total rmse: {:.3f}'.format(rmse_))
            ret['final_rmse'] = rmse_/len(dataloader)

        ret['final_loss'] = running_loss/len(dataloader)

        if classifier is not None:
            ret['accuracy'],ret['class_accuracies'] = classifier.get_final_accuracies()
            ret['report'] = classification_report(y_true,y_pred,target_names=self.class_names)
            ret['confusion_matrix'] = confusion_matrix(y_true,y_pred)
            try:
                ret['roc_auc_score'] = roc_auc_score(y_true,y_pred)
            except:
                pass
        return ret

    def evaluate_food(self,dataloader,metric='accuracy'):
        
        running_loss = 0.
        classifier = None

        classifier = Classifier(self.class_names)

        y_pred = []
        y_true = []

        self.eval()
        rmse_ = 0.
        with torch.no_grad():
            for data_batch in dataloader:
                inputs,labels = data_batch[0],data_batch[1]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(inputs)[0]
                if classifier is not None and metric == 'accuracy':
                    try:
                        classifier.update_accuracies(outputs,labels)
                        y_true.extend(list(labels.squeeze(0).cpu().numpy()))
                        _, preds = torch.max(torch.exp(outputs), 1)
                        y_pred.extend(list(preds.cpu().numpy()))
                    except:
                        pass    
                elif metric == 'rmse':
                    rmse_ += rmse(outputs,labels).cpu().numpy()
            
        self.train()

        ret = {}
        # print('Running_loss: {:.3f}'.format(running_loss))
        if metric == 'rmse':
            print('Total rmse: {:.3f}'.format(rmse_))
            ret['final_rmse'] = rmse_/len(dataloader)

        ret['final_loss'] = running_loss/len(dataloader)

        if classifier is not None:
            ret['accuracy'],ret['class_accuracies'] = classifier.get_final_accuracies()
            ret['report'] = classification_report(y_true,y_pred,target_names=self.class_names)
            ret['confusion_matrix'] = confusion_matrix(y_true,y_pred)
            try:
                ret['roc_auc_score'] = roc_auc_score(y_true,y_pred)
            except:
                pass
        return ret    

    def find_lr(self,trn_loader,init_value=1e-8,final_value=10.,beta=0.98,plot=False):
        
        print('\nFinding the ideal learning rate.')

        model_state = copy.deepcopy(self.model.state_dict())
        optim_state = copy.deepcopy(self.optimizer.state_dict())
        optimizer = self.optimizer
        num = len(trn_loader)-1
        mult = (final_value / init_value) ** (1/num)
        lr = init_value
        optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for data_batch in trn_loader:
            batch_num += 1
            inputs,label1,label2 = data_batch[0],data_batch[1],data_batch[2]
            inputs = inputs.to(self.device)
            label1 = label1.to(self.device)
            label2 = label2.to(self.device)
            labels = (label1,label2)
            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.compute_loss(outputs,labels)[0]
            #Compute the smoothed loss
            if self.parallel:   
                avg_loss = beta * avg_loss + (1-beta) * loss.sum()    
            else:    
                avg_loss = beta * avg_loss + (1-beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                self.log_lrs, self.find_lr_losses = log_lrs,losses
                self.model.load_state_dict(model_state)
                self.optimizer.load_state_dict(optim_state)
                if plot:
                    self.plot_find_lr()
                temp_lr = self.log_lrs[np.argmin(self.find_lr_losses)-(len(self.log_lrs)//8)]
                self.lr = (10**temp_lr)
                print('Found it: {}\n'.format(self.lr))
                return self.lr
            #Record the best loss
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            #Do the SGD step
            if self.parallel:
                loss.sum().backward()
            else:    
                loss.backward()
            optimizer.step()
            #Update the lr for the next step
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr

        self.log_lrs, self.find_lr_losses = log_lrs,losses
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optim_state)
        if plot:
            self.plot_find_lr()
        temp_lr = self.log_lrs[np.argmin(self.find_lr_losses)-(len(self.log_lrs)//10)]
        self.lr = (10**temp_lr)
        print('Found it: {}\n'.format(self.lr))
        return self.lr
            
    def plot_find_lr(self):    
        plt.ylabel("Loss")
        plt.xlabel("Learning Rate (log scale)")
        plt.plot(self.log_lrs,self.find_lr_losses)
        plt.show()

    def classify(self,inputs,thresh = 0.4):#,show = False,mean = None,std = None):
        outputs = self.predict(inputs)
        food,ing = outputs
        try:    
            _, preds = torch.max(torch.exp(food), 1)
        except:
            _, preds = torch.max(torch.exp(food.unsqueeze(0)), 1)
        ing_outs = ing.sigmoid()
        ings = (ing_outs > thresh)
        class_preds = [str(self.class_names[p]) for p in preds]
        ing_preds = [self.ingredeint_names[p.nonzero().squeeze(1).cpu()] for p in ings]
        return class_preds,ing_preds

    def _get_dropout(self):
        return self.dropout_p

    def get_model_params(self):
        params = super(FoodIngredients, self).get_model_params()
        params['class_names'] = self.class_names
        params['num_classes'] = self.num_classes
        params['ingredient_names'] = self.ingredient_names
        params['num_ingredients'] = self.num_ingredients
        params['head1'] = self.head1
        params['head2'] = self.head2
        return params        

class TransferMultiHead(Network):
    def __init__(self,
                concat = False,
                model_name='DenseNet',
                model_type='multi_head',
                lr=0.02,
                optimizer_name = 'Adam',
                criterion1 = nn.CrossEntropyLoss(),
                criterion2 = nn.BCEWithLogitsLoss(),
                loss_w1 = 1.,
                loss_w2 = 1.,
                dropout_p = 0.45,
                pretrained = True,
                device = None,
                best_accuracy = 0.,
                best_validation_loss = None,
                best_model_file ='best_model.pth',
                kornia_transforms = None,
                head1 = {'num_outputs':10,
                'layers':[],
                'model_type':'classifier'
                },
                head2 = {'num_outputs':10,
                'layers':[],
                'model_type':'multi_label_classifier'
                },
                channels = 3,
                class_names = [],
                num_classes = None,
                multi_names = [],
                num_multi = None,
                add_extra = True,
                set_params = True,
                set_head = True
                ):
        super().__init__(device=device)
        self.train_parameters = []
        if kornia_transforms is not None:
            self.kornia_transforms = nn.Sequential(*kornia_transforms)
            self.train_parameters += list(self.kornia_transforms.parameters())
        else:
            self.kornia_transforms = None
        efficient_nets = ['efficientnet-b0','efficientnet-b1','efficientnet-b2',
        'efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6','efficientnet-b7']
        if model_name.lower() in efficient_nets:
            # set_head = False
            self.model = EfficientNet.from_pretrained(model_name,num_classes,channels)
        else:    
            self.set_transfer_model(model_name,pretrained=pretrained,add_extra=add_extra,dropout_p=dropout_p)

        if set_head:
            self.set_model_head(model_name = model_name,
                    concat = concat,
                    head1 = head1,
                    head2 = head2,
                    channels = channels,
                    dropout_p = dropout_p,
                    criterion1 = criterion1,
                    criterion2 = criterion2,
                    device = device
                )
        self.train_parameters += list(self.model.parameters()) + list(self.header.parameters())
        if set_params:
            self.set_model_params(
                                params = self.train_parameters,
                                criterion1 = criterion1,
                                criterion2 = criterion2,
                                optimizer_name = optimizer_name,
                                lr = lr,
                                dropout_p = dropout_p,
                                model_name = model_name,
                                model_type = model_type,
                                best_accuracy = best_accuracy,
                                best_validation_loss = best_validation_loss,
                                best_model_file = best_model_file,
                                class_names = class_names,
                                num_classes = num_classes,
                                multi_names = multi_names, 
                                num_multi = num_multi,
                                channels = channels,
                                loss_w1 = loss_w1,
                                loss_w2 = loss_w2
                                )
        self.to(device)

    def set_model_params(self,
                            params = None,
                            criterion1 = nn.CrossEntropyLoss(),
                            criterion2 = nn.BCEWithLogitsLoss(),
                            loss_w1=1.,
                            loss_w2=1.,
                            optimizer_name = 'Adam',
                            lr  = 0.1,
                            dropout_p = 0.45,
                            model_name = 'DenseNet',
                            model_type = 'cv_transfer',
                            best_accuracy = 0.,
                            best_validation_loss = None,
                            best_model_file = 'best_model_file.pth',
                            head1 = {'num_outputs':10,
                                'layers':[],
                                'model_type':'classifier'
                                },
                            head2 = {'num_outputs':10,
                                'layers':[],
                                'model_type':'muilti_label_classifier'
                                },       
                            class_names = [],
                            num_classes = None,
                            multi_names = [],
                            num_multi = None,
                            channels = 3):
        
        # print('Emotions: current best accuracy = {:.3f}'.format(best_accuracy))
        # if best_validation_loss is not None:
        #     print('Multi Emotions: current best loss = {:.3f}'.format(best_validation_loss))

        super(TransferMultiHead, self).set_model_params(
                                                params = params,
                                                optimizer_name = optimizer_name,
                                                lr = lr,
                                                dropout_p = dropout_p,
                                                model_name = model_name,
                                                model_type = model_type,
                                                best_accuracy = best_accuracy,
                                                best_validation_loss = best_validation_loss,
                                                best_model_file = best_model_file
                                                )
        self.class_names = class_names
        self.num_classes = num_classes                                              
        self.multi_names = multi_names
        self.num_multi = num_multi
        self.channels = channels
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.loss_w1 = loss_w1
        self.loss_w2 = loss_w2

    def forward(self,x):
        # l = list(self.model.children())
        # for m in l[:-2]:
        #     x = m(x)
        # clas = l[-2](x)
        # multi_class = l[-1](x)
        # return (clas,multi_class)
        if self.kornia_transforms is not None:
            x = self.kornia_transforms(x)
        return self.header(self.model(x))
    
    def compute_loss(self,outputs,labels): 
        out1,out2 = outputs
        label1,label2 = labels
        loss1 = self.criterion1(out1,label1)
        loss2 = self.criterion2(out2,label2)
        # print(loss1,loss2)
        return [(loss1 * self.loss_w1)+(loss2 * self.loss_w2)]

    def freeze(self,train_classifier=True):
        super(TransferMultiHead, self).freeze()
        if train_classifier:
            for param in self.model.fc1.parameters():
                    param.requires_grad = True
            for param in self.model.fc2.parameters():
                    param.requires_grad = True     

    def parallelize(self):
        self.parallel = True
        self.model = DataParallelModel(self.model)
        self.criterion  = DataParallelCriterion(self.criterion)

    def set_transfer_model(self,mname,pretrained=True,add_extra=True,dropout_p = 0.45):   
        self.model = None
        models_dict = {

            'densenet': {'model':densenet121(pretrained=pretrained),'conv_channels':1024},
            'resnet34': {'model':resnet34(pretrained=pretrained),'conv_channels':512},
            'resnet50': {'model':resnet50(pretrained=pretrained),'conv_channels':2048},
            'resnext50': {'model':resnext50_32x4d(pretrained=pretrained),'conv_channels':2048},
            'resnext101': {'model':resnext101_32x8d(pretrained=pretrained),'conv_channels':2048}

        }
        meta = models_dict[mname.lower()]
        try:
            model = meta['model']
            for param in model.parameters():
                param.requires_grad = False
            self.model = model    
            print('Setting transfer learning model: self.model set to {}'.format(mname))
        except:
            print('Setting transfer learning model: model name {} not supported'.format(mname))            

        # creating and adding extra layers to the model
        dream_model = None
        if add_extra:
            channels = meta['conv_channels']
            dream_model = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,1),
                # Printer(),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p),
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p),
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p)
                )        
        self.dream_model = dream_model          
    
    def create_head(self,head,fc,last_fc = False):
        try:
            in_features =  fc.in_features
        except:
            in_features = fc.model.out.in_features 
        if last_fc:
            in_features+=1
        fc = FC(
                num_inputs = in_features,
                num_outputs = head['num_outputs'],
                layers = head['layers'],
                model_type = head['model_type'],
                output_non_linearity = head['output_non_linearity'],
                dropout_p = head['dropout_p'],
                criterion = head['criterion'],
                optimizer_name = None,
                device = head['device']
                )
        return fc

    def set_model_head(self,
                        model_name = 'DenseNet',
                        concat = False,
                        head1 = {'num_outputs':10,
                                'layers':[],
                                'class_names': None,
                                'model_type':'classifier'
                                },
                        head2 = {'num_outputs':10,
                                'layers':[],
                                'class_names': None,
                                'model_type':'muilti_label_classifier'
                                },
                        channels = 3,
                        criterion1 = nn.CrossEntropyLoss(),
                        criterion2 = nn.BCEWithLogitsLoss(), 
                        adaptive = True,       
                        dropout_p = 0.45,
                        device = None):

        models_meta = {
        'resnet34': {'conv_channels':512,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'resnet50': {'conv_channels':2048,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'resnext50': {'conv_channels':2048,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'resnext101': {'conv_channels':2048,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'densenet': {'conv_channels':1024,'head_id': -1,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool]
                    ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1)]}
        }
        heads = [head1,head2]
        crits = [criterion1,criterion2]
        if hasattr(self.model,'_fc'):
            model_end = self.model._fc
        else:
            name = model_name.lower()
            meta = models_meta[name]
            modules = list(self.model.children())
            l = modules[:meta['head_id']]
            if self.dream_model:
                l+=self.dream_model
            model_end = modules[-1]
            if adaptive:
                l += meta['adaptive_head']
            else:
                l += meta['normal_head']
            if channels == 1:
                l.insert(0,nn.Conv2d(1,3,3,1,1))
            model = nn.Sequential(*l)
            self.model = model
        fcs = []
        last_fc = False
        for head,criterion in zip(heads,crits):
            head['device'] = device
            head['dropout_p'] = dropout_p
            head['criterion'] = criterion
            if 'output_non_linearity' not in head.keys():
                head['output_non_linearity'] = None
            if len(fcs) == 1:
                last_fc = True
            fc = self.create_head(head,model_end,last_fc)
            fcs.append(fc)          
        if concat:
            self.header = MultiConcatHeader(fcs[0],fcs[1])
        else:
            self.header = MultiSeparateHeader(fcs[0],fcs[1])
        self.head1 = head1
        self.head2 = head2
        
        print('Multi-head set up complete.')

    def train_(self,e,trainloader,optimizer,print_every,clip=False):

        epoch,epochs = e
        self.train()
        t0 = time.time()
        t1 = time.time()
        batches = 0
        running_loss = 0.
        for data_batch in trainloader:
            inputs,label1,label2 = data_batch[0],data_batch[1],data_batch[2]
            batches += 1
            inputs = inputs.to(self.device)
            label1 = label1.to(self.device)
            label2 = label2.to(self.device)
            labels = (label1,label2)
            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.compute_loss(outputs,labels)[0]
            if self.parallel:
                loss.sum().backward()
                loss = loss.sum()
            else:    
                loss.backward()
                loss = loss.item()
            if clip:
                nn.utils.clip_grad_norm(self.train_parameters,1.)
            optimizer.step()
            running_loss += loss
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
                        f"Batch training loss: {loss:.3f}\n"
                        f"Average training loss: {running_loss/(batches):.3f}\n"
                        '+----------------------------------------------------------------------+\n'     
                        )
                t0 = time.time()
        return running_loss/len(trainloader) 

    def evaluate(self,dataloader,metric='accuracy'):
        
        running_loss = 0.
        classifier = None

        if self.model_type == 'classifier':# or self.num_classes is not None:
            classifier = Classifier(self.class_names)

        y_pred = []
        y_true = []

        self.eval()
        rmse_ = 0.
        with torch.no_grad():
            for data_batch in dataloader:
                inputs,label1,label2 = data_batch[0],data_batch[1],data_batch[2]
                inputs = inputs.to(self.device)
                label1 = label1.to(self.device)
                label2 = label2.to(self.device)
                labels = (label1,label2)
                outputs = self.forward(inputs)
                loss = self.compute_loss(outputs,labels)[0]
                if self.parallel:
                    running_loss += loss.sum()
                    outputs = parallel.gather(outputs,self.device)
                else:        
                    running_loss += loss.item()
                if classifier is not None and metric == 'accuracy':
                    classifier.update_accuracies(outputs,labels)
                    y_true.extend(list(labels.squeeze(0).cpu().numpy()))
                    _, preds = torch.max(torch.exp(outputs), 1)
                    y_pred.extend(list(preds.cpu().numpy()))
                elif metric == 'rmse':
                    rmse_ += rmse(outputs,labels).cpu().numpy()
            
        self.train()

        ret = {}
        # print('Running_loss: {:.3f}'.format(running_loss))
        if metric == 'rmse':
            print('Total rmse: {:.3f}'.format(rmse_))
            ret['final_rmse'] = rmse_/len(dataloader)

        ret['final_loss'] = running_loss/len(dataloader)

        if classifier is not None:
            ret['accuracy'],ret['class_accuracies'] = classifier.get_final_accuracies()
            ret['report'] = classification_report(y_true,y_pred,target_names=self.class_names)
            ret['confusion_matrix'] = confusion_matrix(y_true,y_pred)
            try:
                ret['roc_auc_score'] = roc_auc_score(y_true,y_pred)
            except:
                pass
        return ret

    def evaluate_multi_head(self,dataloader,metric='accuracy'):
        
        running_loss = 0.
        classifier = None

        classifier = Classifier(self.class_names)

        y_pred = []
        y_true = []

        self.eval()
        rmse_ = 0.
        with torch.no_grad():
            for data_batch in dataloader:
                inputs,labels = data_batch[0],data_batch[1]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(inputs)[0]
                if classifier is not None and metric == 'accuracy':
                    try:
                        classifier.update_accuracies(outputs,labels)
                        y_true.extend(list(labels.squeeze(0).cpu().numpy()))
                        _, preds = torch.max(torch.exp(outputs), 1)
                        y_pred.extend(list(preds.cpu().numpy()))
                    except:
                        pass    
                elif metric == 'rmse':
                    rmse_ += rmse(outputs,labels).cpu().numpy()
            
        self.train()

        ret = {}
        # print('Running_loss: {:.3f}'.format(running_loss))
        if metric == 'rmse':
            print('Total rmse: {:.3f}'.format(rmse_))
            ret['final_rmse'] = rmse_/len(dataloader)

        ret['final_loss'] = running_loss/len(dataloader)

        if classifier is not None:
            ret['accuracy'],ret['class_accuracies'] = classifier.get_final_accuracies()
            ret['report'] = classification_report(y_true,y_pred,target_names=self.class_names)
            ret['confusion_matrix'] = confusion_matrix(y_true,y_pred)
            try:
                ret['roc_auc_score'] = roc_auc_score(y_true,y_pred)
            except:
                pass
        return ret    

    # def classify(self,inputs,thresh = 0.4,class_names = None):
    #     if class_names is None:
    #         class_names = self.class_names
    #     outputs = self.predict(inputs)[0]
    #     if self.model_type == 'classifier':
    #         try:    
    #             _, preds = torch.max(torch.exp(outputs), 1)
    #         except:
    #             _, preds = torch.max(torch.exp(outputs.unsqueeze(0)), 1)
    #     else:
    #         outputs = outputs.sigmoid()
    #         preds = (outputs >= thresh).nonzero().squeeze(1)
    #     class_preds = [str(class_names[p]) for p in preds]
    #     # imgs = batch_to_imgs(inputs.cpu(),mean,std)
    #     # if show:
    #         # plot_in_row(imgs,titles=class_preds)
    #     return class_preds
    
    def classify(self,inputs,multi = True,thresh = 0.4):#,show = False,mean = None,std = None):
        outputs = self.predict(inputs)
        clas,multi_class = outputs
        try:    
            _, preds = torch.max(torch.exp(clas), 1)
        except:
            _, preds = torch.max(torch.exp(clas.unsqueeze(0)), 1)
        class_preds = [str(self.class_names[p]) for p in preds]        
        if multi:
            multi_out = multi_class.sigmoid()
            multi_outs = (multi_out > thresh)
            multi_preds = [self.multi_names[p.nonzero().squeeze(1).cpu()] for p in multi_outs]
        else:
            multi_preds = []    
        return class_preds,multi_preds

    def find_lr(self,trn_loader,init_value=1e-8,final_value=10.,beta=0.98,plot=False):
        
        print('\nFinding the ideal learning rate.')

        model_state = copy.deepcopy(self.model.state_dict())
        optim_state = copy.deepcopy(self.optimizer.state_dict())
        optimizer = self.optimizer
        num = len(trn_loader)-1
        mult = (final_value / init_value) ** (1/num)
        lr = init_value
        optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for data_batch in trn_loader:
            batch_num += 1
            if batch_num % 100 == 0:
                print(batch_num)
            inputs,label1,label2 = data_batch[0],data_batch[1],data_batch[2]
            inputs = inputs.to(self.device)
            label1 = label1.to(self.device)
            label2 = label2.to(self.device)
            labels = (label1,label2)
            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.compute_loss(outputs,labels)[0]
            #Compute the smoothed loss
            if self.parallel:   
                avg_loss = beta * avg_loss + (1-beta) * loss.sum()    
            else:    
                avg_loss = beta * avg_loss + (1-beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                self.log_lrs, self.find_lr_losses = log_lrs,losses
                self.model.load_state_dict(model_state)
                self.optimizer.load_state_dict(optim_state)
                if plot:
                    self.plot_find_lr()
                temp_lr = self.log_lrs[np.argmin(self.find_lr_losses)-(len(self.log_lrs)//8)]
                self.lr = (10**temp_lr)
                print('Found it: {}\n'.format(self.lr))
                return self.lr
            #Record the best loss
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            #Do the SGD step
            if self.parallel:
                loss.sum().backward()
            else:    
                loss.backward()
            optimizer.step()
            #Update the lr for the next step
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr

        self.log_lrs, self.find_lr_losses = log_lrs,losses
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optim_state)
        if plot:
            self.plot_find_lr()
        temp_lr = self.log_lrs[np.argmin(self.find_lr_losses)-(len(self.log_lrs)//10)]
        self.lr = (10**temp_lr)
        print('Found it: {}\n'.format(self.lr))
        return self.lr
            
    def plot_find_lr(self):    
        plt.ylabel("Loss")
        plt.xlabel("Learning Rate (log scale)")
        plt.plot(self.log_lrs,self.find_lr_losses)
        plt.show()

    def _get_dropout(self):
        return self.dropout_p

    def get_model_params(self):
        params = super(TransferMultiHead, self).get_model_params()
        params['class_names'] = self.class_names
        params['num_classes'] = self.num_classes
        params['multi_names'] = self.multi_names
        params['num_multi'] = self.num_multi
        params['head1'] = self.head1
        params['head2'] = self.head2
        return params 

class TransferMultiHeadOld(Network):
    def __init__(self,
                model_name='DenseNet',
                model_type='multi_head',
                lr=0.02,
                optimizer_name = 'Adam',
                criterion1 = nn.CrossEntropyLoss(),
                criterion2 = nn.BCEWithLogitsLoss(),
                dropout_p=0.45,
                pretrained=True,
                device=None,
                best_accuracy=0.,
                best_validation_loss=None,
                best_model_file ='best_model.pth',
                head1 = {'num_outputs':10,
                'layers':[],
                'model_type':'classifier'
                },
                head2 = {'num_outputs':10,
                'layers':[],
                'model_type':'multi_label_classifier'
                },
                channels = 3,
                class_names = [],
                num_classes = None,
                multi_names = [],
                num_multi = None,
                add_extra = True,
                set_params = True,
                set_head = True
                ):

        super().__init__(device=device)

        self.set_transfer_model(model_name,pretrained=pretrained,add_extra=add_extra,dropout_p=dropout_p)

        if set_head:
            self.set_model_head(model_name = model_name,
                    head1 = head1,
                    head2 = head2,
                    channels = channels,
                    dropout_p = dropout_p,
                    criterion1 = criterion1,
                    criterion2 = criterion2,
                    device = device
                )
        if set_params:
            self.set_model_params(
                                optimizer_name = optimizer_name,
                                lr = lr,
                                dropout_p = dropout_p,
                                model_name = model_name,
                                model_type = model_type,
                                best_accuracy = best_accuracy,
                                best_validation_loss = best_validation_loss,
                                best_model_file = best_model_file,
                                class_names = class_names,
                                num_classes = num_classes,
                                multi_names = multi_names, 
                                num_multi = num_multi,
                                channels = channels
                                )

        self.model = self.model.to(device)
        
    def set_model_params(self,
                            criterion1 = nn.CrossEntropyLoss(),
                            criterion2 = nn.BCEWithLogitsLoss(),
                            optimizer_name = 'Adam',
                            lr  = 0.1,
                            dropout_p = 0.45,
                            model_name = 'DenseNet',
                            model_type = 'cv_transfer',
                            best_accuracy = 0.,
                            best_validation_loss = None,
                            best_model_file = 'best_model_file.pth',
                            head1 = {'num_outputs':10,
                                'layers':[],
                                'model_type':'classifier'
                                },
                            head2 = {'num_outputs':10,
                                'layers':[],
                                'model_type':'muilti_label_classifier'
                                },       
                            class_names = [],
                            num_classes = None,
                            multi_names = [],
                            num_multi = None,
                            channels = 3):
        
        # print('Emotions: current best accuracy = {:.3f}'.format(best_accuracy))
        # if best_validation_loss is not None:
        #     print('Multi Emotions: current best loss = {:.3f}'.format(best_validation_loss))

        
        super(TransferMultiHeadOld, self).set_model_params(
                                                optimizer_name = optimizer_name,
                                                lr = lr,
                                                dropout_p = dropout_p,
                                                model_name = model_name,
                                                model_type = model_type,
                                                best_accuracy = best_accuracy,
                                                best_validation_loss = best_validation_loss,
                                                best_model_file = best_model_file
                                                )
        self.class_names = class_names
        self.num_classes = num_classes                                              
        self.multi_names = multi_names
        self.num_multi = num_multi
        self.channels = channels
        self.criterion1 = criterion1
        self.criterion2 = criterion2

    def forward(self,x):
        l = list(self.model.children())
        for m in l[:-2]:
            x = m(x)
        clas = l[-2](x)
        multi_class = l[-1](x)
        return (clas,multi_class)
    
    def compute_loss(self,outputs,labels,w1 = 1.,w2 = 1.): 
        out1,out2 = outputs
        label1,label2 = labels
        loss1 = self.criterion1(out1,label1)
        loss2 = self.criterion2(out2,label2)
        # print(loss1,loss2)
        return [(loss1*w1)+(loss2*w2)]

    def freeze(self,train_classifier=True):
        super(TransferMultiHeadOld, self).freeze()
        if train_classifier:
            for param in self.model.fc1.parameters():
                    param.requires_grad = True
            for param in self.model.fc2.parameters():
                    param.requires_grad = True     

    def parallelize(self):
        self.parallel = True
        self.model = DataParallelModel(self.model)
        self.criterion  = DataParallelCriterion(self.criterion)

    def set_transfer_model(self,mname,pretrained=True,add_extra=True,dropout_p = 0.45):   
        self.model = None
        models_dict = {

            'densenet': {'model':densenet121(pretrained=pretrained),'conv_channels':1024},
            'resnet34': {'model':resnet34(pretrained=pretrained),'conv_channels':512},
            'resnet50': {'model':resnet50(pretrained=pretrained),'conv_channels':2048},
            'resnext50': {'model':resnext50_32x4d(pretrained=pretrained),'conv_channels':2048},
            'resnext101': {'model':resnext101_32x8d(pretrained=pretrained),'conv_channels':2048}

        }
        meta = models_dict[mname.lower()]
        try:
            model = meta['model']
            for param in model.parameters():
                param.requires_grad = False
            self.model = model    
            print('Setting transfer learning model: self.model set to {}'.format(mname))
        except:
            print('Setting transfer learning model: model name {} not supported'.format(mname))            

        # creating and adding extra layers to the model
        dream_model = None
        if add_extra:
            channels = meta['conv_channels']
            dream_model = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,1),
                # Printer(),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p),
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p),
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p)
                )        
        self.dream_model = dream_model          
            
    def set_model_head(self,
                        model_name = 'DenseNet',
                        head1 = {'num_outputs':10,
                                'layers':[],
                                'class_names': None,
                                'model_type':'classifier'
                                },
                        head2 = {'num_outputs':10,
                                'layers':[],
                                'class_names': None,
                                'model_type':'muilti_label_classifier'
                                },
                        channels = 3,
                        criterion1 = nn.CrossEntropyLoss(),
                        criterion2 = nn.BCEWithLogitsLoss(), 
                        adaptive = True,       
                        dropout_p = 0.45,
                        device = None):

        models_meta = {
        'resnet34': {'conv_channels':512,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'resnet50': {'conv_channels':2048,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'resnext50': {'conv_channels':2048,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'resnext101': {'conv_channels':2048,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'densenet': {'conv_channels':1024,'head_id': -1,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool]
                    ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1)]}
        }

        name = model_name.lower()
        meta = models_meta[name]
        modules = list(self.model.children())
        l = modules[:meta['head_id']]
        if self.dream_model:
            l+=self.dream_model
        heads = [head1,head2]
        crits = [criterion1,criterion2]    
        fcs = []
        for head,criterion in zip(heads,crits):
            head['criterion'] = criterion
            if 'output_non_linearity' not in head.keys():
                head['output_non_linearity'] = None
            fc = modules[-1]
            try:
                in_features =  fc.in_features
            except:
                in_features = fc.model.out.in_features    
            fc = FC(
                    num_inputs = in_features,
                    num_outputs = head['num_outputs'],
                    layers = head['layers'],
                    model_type = head['model_type'],
                    output_non_linearity = head['output_non_linearity'],
                    dropout_p = dropout_p,
                    criterion = head['criterion'],
                    optimizer_name = None,
                    device = device
                    )
            fcs.append(fc)          
        if adaptive:
            l += meta['adaptive_head']
        else:
            l += meta['normal_head']
        if channels == 1:
            l.insert(0,nn.Conv2d(1,3,3,1,1))
        model = nn.Sequential(*l)
        model.add_module('fc1',fcs[0])
        model.add_module('fc2',fcs[1])
        self.model = model
        self.head1 = head1
        self.head2 = head2
        
        print('Multi-head set up complete.')

    def train_(self,e,trainloader,optimizer,print_every):

        epoch,epochs = e
        self.train()
        t0 = time.time()
        t1 = time.time()
        batches = 0
        running_loss = 0.
        for data_batch in trainloader:
            inputs,label1,label2 = data_batch[0],data_batch[1],data_batch[2]
            batches += 1
            inputs = inputs.to(self.device)
            label1 = label1.to(self.device)
            label2 = label2.to(self.device)
            labels = (label1,label2)
            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.compute_loss(outputs,labels)[0]
            if self.parallel:
                loss.sum().backward()
                loss = loss.sum()
            else:    
                loss.backward()
                loss = loss.item()
            optimizer.step()
            running_loss += loss
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
                        f"Batch training loss: {loss:.3f}\n"
                        f"Average training loss: {running_loss/(batches):.3f}\n"
                        '+----------------------------------------------------------------------+\n'     
                        )
                t0 = time.time()
        return running_loss/len(trainloader) 

    def evaluate(self,dataloader,metric='accuracy'):
        
        running_loss = 0.
        classifier = None

        if self.model_type == 'classifier':# or self.num_classes is not None:
            classifier = Classifier(self.class_names)

        y_pred = []
        y_true = []

        self.eval()
        rmse_ = 0.
        with torch.no_grad():
            for data_batch in dataloader:
                inputs,label1,label2 = data_batch[0],data_batch[1],data_batch[2]
                inputs = inputs.to(self.device)
                label1 = label1.to(self.device)
                label2 = label2.to(self.device)
                labels = (label1,label2)
                outputs = self.forward(inputs)
                loss = self.compute_loss(outputs,labels)[0]
                if self.parallel:
                    running_loss += loss.sum()
                    outputs = parallel.gather(outputs,self.device)
                else:        
                    running_loss += loss.item()
                if classifier is not None and metric == 'accuracy':
                    classifier.update_accuracies(outputs,labels)
                    y_true.extend(list(labels.squeeze(0).cpu().numpy()))
                    _, preds = torch.max(torch.exp(outputs), 1)
                    y_pred.extend(list(preds.cpu().numpy()))
                elif metric == 'rmse':
                    rmse_ += rmse(outputs,labels).cpu().numpy()
            
        self.train()

        ret = {}
        # print('Running_loss: {:.3f}'.format(running_loss))
        if metric == 'rmse':
            print('Total rmse: {:.3f}'.format(rmse_))
            ret['final_rmse'] = rmse_/len(dataloader)

        ret['final_loss'] = running_loss/len(dataloader)

        if classifier is not None:
            ret['accuracy'],ret['class_accuracies'] = classifier.get_final_accuracies()
            ret['report'] = classification_report(y_true,y_pred,target_names=self.class_names)
            ret['confusion_matrix'] = confusion_matrix(y_true,y_pred)
            try:
                ret['roc_auc_score'] = roc_auc_score(y_true,y_pred)
            except:
                pass
        return ret

    def evaluate_multi_head(self,dataloader,metric='accuracy'):
        
        running_loss = 0.
        classifier = None

        classifier = Classifier(self.class_names)

        y_pred = []
        y_true = []

        self.eval()
        rmse_ = 0.
        with torch.no_grad():
            for data_batch in dataloader:
                inputs,labels = data_batch[0],data_batch[1]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(inputs)[0]
                if classifier is not None and metric == 'accuracy':
                    try:
                        classifier.update_accuracies(outputs,labels)
                        y_true.extend(list(labels.squeeze(0).cpu().numpy()))
                        _, preds = torch.max(torch.exp(outputs), 1)
                        y_pred.extend(list(preds.cpu().numpy()))
                    except:
                        pass    
                elif metric == 'rmse':
                    rmse_ += rmse(outputs,labels).cpu().numpy()
            
        self.train()

        ret = {}
        # print('Running_loss: {:.3f}'.format(running_loss))
        if metric == 'rmse':
            print('Total rmse: {:.3f}'.format(rmse_))
            ret['final_rmse'] = rmse_/len(dataloader)

        ret['final_loss'] = running_loss/len(dataloader)

        if classifier is not None:
            ret['accuracy'],ret['class_accuracies'] = classifier.get_final_accuracies()
            ret['report'] = classification_report(y_true,y_pred,target_names=self.class_names)
            ret['confusion_matrix'] = confusion_matrix(y_true,y_pred)
            try:
                ret['roc_auc_score'] = roc_auc_score(y_true,y_pred)
            except:
                pass
        return ret    

    # def classify(self,inputs,thresh = 0.4,class_names = None):
    #     if class_names is None:
    #         class_names = self.class_names
    #     outputs = self.predict(inputs)[0]
    #     if self.model_type == 'classifier':
    #         try:    
    #             _, preds = torch.max(torch.exp(outputs), 1)
    #         except:
    #             _, preds = torch.max(torch.exp(outputs.unsqueeze(0)), 1)
    #     else:
    #         outputs = outputs.sigmoid()
    #         preds = (outputs >= thresh).nonzero().squeeze(1)
    #     class_preds = [str(class_names[p]) for p in preds]
    #     # imgs = batch_to_imgs(inputs.cpu(),mean,std)
    #     # if show:
    #         # plot_in_row(imgs,titles=class_preds)
    #     return class_preds
    
    def classify(self,inputs,multi = True,thresh = 0.4):#,show = False,mean = None,std = None):
        outputs = self.predict(inputs)
        clas,multi_class = outputs
        try:    
            _, preds = torch.max(torch.exp(clas), 1)
        except:
            _, preds = torch.max(torch.exp(clas.unsqueeze(0)), 1)
        class_preds = [str(self.class_names[p]) for p in preds]        
        if multi:
            multi_out = multi_class.sigmoid()
            multi_outs = (multi_out > thresh)
            multi_preds = [self.multi_names[p.nonzero().squeeze(1).cpu()] for p in multi_outs]
        else:
            multi_preds = []    
        return class_preds,multi_preds

    def find_lr(self,trn_loader,init_value=1e-8,final_value=10.,beta=0.98,plot=False):
        
        print('\nFinding the ideal learning rate.')

        model_state = copy.deepcopy(self.model.state_dict())
        optim_state = copy.deepcopy(self.optimizer.state_dict())
        optimizer = self.optimizer
        num = len(trn_loader)-1
        mult = (final_value / init_value) ** (1/num)
        lr = init_value
        optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for data_batch in trn_loader:
            batch_num += 1
            if batch_num % 100 == 0:
                print(batch_num)
            inputs,label1,label2 = data_batch[0],data_batch[1],data_batch[2]
            inputs = inputs.to(self.device)
            label1 = label1.to(self.device)
            label2 = label2.to(self.device)
            labels = (label1,label2)
            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.compute_loss(outputs,labels)[0]
            #Compute the smoothed loss
            if self.parallel:   
                avg_loss = beta * avg_loss + (1-beta) * loss.sum()    
            else:    
                avg_loss = beta * avg_loss + (1-beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                self.log_lrs, self.find_lr_losses = log_lrs,losses
                self.model.load_state_dict(model_state)
                self.optimizer.load_state_dict(optim_state)
                if plot:
                    self.plot_find_lr()
                temp_lr = self.log_lrs[np.argmin(self.find_lr_losses)-(len(self.log_lrs)//8)]
                self.lr = (10**temp_lr)
                print('Found it: {}\n'.format(self.lr))
                return self.lr
            #Record the best loss
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            #Do the SGD step
            if self.parallel:
                loss.sum().backward()
            else:    
                loss.backward()
            optimizer.step()
            #Update the lr for the next step
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr

        self.log_lrs, self.find_lr_losses = log_lrs,losses
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optim_state)
        if plot:
            self.plot_find_lr()
        temp_lr = self.log_lrs[np.argmin(self.find_lr_losses)-(len(self.log_lrs)//10)]
        self.lr = (10**temp_lr)
        print('Found it: {}\n'.format(self.lr))
        return self.lr
            
    def plot_find_lr(self):    
        plt.ylabel("Loss")
        plt.xlabel("Learning Rate (log scale)")
        plt.plot(self.log_lrs,self.find_lr_losses)
        plt.show()

    def _get_dropout(self):
        return self.dropout_p

    def get_model_params(self):
        params = super(TransferMultiHeadOld, self).get_model_params()
        params['class_names'] = self.class_names
        params['num_classes'] = self.num_classes
        params['multi_names'] = self.multi_names
        params['num_multi'] = self.num_multi
        params['head1'] = self.head1
        params['head2'] = self.head2
        return params        

class TransferNetworkImg(Network):
    def __init__(self,
                 model_name='DenseNet',
                 model_type='cv_transfer',
                 lr=0.02,
                 criterion = nn.CrossEntropyLoss(),
                 optimizer_name = 'Adam',
                 dropout_p=0.45,
                 pretrained=True,
                 device=None,
                 best_accuracy=0.,
                 best_validation_loss=None,
                 best_model_file ='best_model.pth',
                 head = {'num_outputs':10,
                    'layers':[],
                    'model_type':'classifier'
                 },
                 kornia_transforms = None,
                 channels = 3,
                 class_names = [],
                 num_classes = None,
                 add_extra = True,
                 set_params = True,
                 set_head = True
                 ):

        super().__init__(device=device)
        model_modules = []
        if kornia_transforms is not None:
            self.kornia_transforms = nn.Sequential(*kornia_transforms)
            model_modules.append(self.kornia_transforms)
        else:
            self.kornia_transforms = None
        efficient_nets = ['efficientnet-b0','efficientnet-b1','efficientnet-b2',
        'efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6','efficientnet-b7']
        if model_name.lower() in efficient_nets:
            # set_head = False
            self.backbone = EfficientNet.from_pretrained(model_name,num_classes,channels)
        else:    
            self.set_transfer_model(model_name,pretrained=pretrained,add_extra=add_extra,dropout_p=dropout_p)

        if set_head:
            self.set_model_head(model_name = model_name,
                    head = head,
                    channels = channels,
                    dropout_p = dropout_p,
                    criterion = criterion,
                    device = device
                )
        model_modules += [self.backbone,self.header]
        self.model = nn.Sequential(*model_modules)
        if set_params:
            self.set_model_params(
                              criterion = criterion,
                              optimizer_name = optimizer_name,
                              lr = lr,
                              dropout_p = dropout_p,
                              model_name = model_name,
                              model_type = model_type,
                              best_accuracy = best_accuracy,
                              best_validation_loss = best_validation_loss,
                              best_model_file = best_model_file,
                              class_names = class_names,
                              num_classes = num_classes,
                              channels = channels
                              )

        self.model = self.model.to(device)
        
    def set_model_params(self,
                         params = None,
                         criterion = nn.CrossEntropyLoss(),
                         optimizer_name = 'Adam',
                         lr  = 0.1,
                         dropout_p = 0.45,
                         model_name = 'DenseNet',
                         model_type = 'cv_transfer',
                         best_accuracy = 0.,
                         best_validation_loss = None,
                         best_model_file = 'best_model_file.pth',
                         class_names = [],
                         num_classes = None,
                         channels = 3):
        
        # print('Transfer Learning: current best accuracy = {:.3f}'.format(best_accuracy))
        
        super(TransferNetworkImg, self).set_model_params(
                                              params = params,
                                              criterion = criterion,
                                              optimizer_name = optimizer_name,
                                              lr = lr,
                                              dropout_p = dropout_p,
                                              model_name = model_name,
                                              model_type = model_type,
                                              best_accuracy = best_accuracy,
                                              best_validation_loss = best_validation_loss,
                                              best_model_file = best_model_file
                                              )
        self.class_names = class_names
        self.num_classes = num_classes
        self.channels = channels                                              
        if len(class_names) == 0:
            self.class_names = {k:str(v) for k,v in enumerate(list(range(self.head['num_outputs'])))}

    def forward(self,x):
        return self.model(x)
    
    def freeze(self,idx=None):
        if idx is None:
            super(TransferNetworkImg, self).freeze()
        else:
            for p in self.model[idx].parameters():
                p.requires_grad = False

    def parallelize(self):
        self.parallel = True
        self.model = DataParallelModel(self.model)
        self.criterion  = DataParallelCriterion(self.criterion)

    def set_transfer_model(self,mname,pretrained=True,add_extra=True,dropout_p = 0.45):   
        self.backbone = None
        models_dict = {

            'densenet': {'model':densenet121,'conv_channels':1024},
            'resnet34': {'model':resnet34,'conv_channels':512},
            'resnet50': {'model':resnet50,'conv_channels':2048},
            'resnext50': {'model':resnext50_32x4d,'conv_channels':2048},
            'resnext101': {'model':resnext101_32x8d,'conv_channels':2048}

        }
        meta = models_dict[mname.lower()]
        try:
            model = meta['model'](pretrained=pretrained)
            for param in model.parameters():
                param.requires_grad = False
            self.backbone = model    
            print('Setting transfer learning model: self.model set to {}'.format(mname))
        except:
            print('Setting transfer learning model: model name {} not supported'.format(mname))            

        # creating and adding extra layers to the model
        dream_model = None
        if add_extra:
            channels = meta['conv_channels']
            dream_model = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,1),
                # Printer(),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p),
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p),
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p)
                )        
        self.dream_model = dream_model          
    
    def create_head(self,head,fc):
        try:
            in_features =  fc.in_features
        except:
            in_features = fc.model.out.in_features    
        fc = FC(
                num_inputs = in_features,
                num_outputs = head['num_outputs'],
                layers = head['layers'],
                model_type = head['model_type'],
                output_non_linearity = head['output_non_linearity'],
                dropout_p = head['dropout_p'],
                criterion = head['criterion'],
                optimizer_name = None,
                device = head['device']
                )
        return fc

    def set_model_head(self,
                        model_name = 'DenseNet',
                        head = {'num_outputs':10,
                                'layers':[],
                                'class_names': None,
                                'model_type':'classifier'
                               },
                        channels = 3,
                        criterion = nn.NLLLoss(),  
                        adaptive = True,       
                        dropout_p = 0.45,
                        device = None):

        models_meta = {
        'resnet34': {'conv_channels':512,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'resnet50': {'conv_channels':2048,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'resnext50': {'conv_channels':2048,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'resnext101': {'conv_channels':2048,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'densenet': {'conv_channels':1024,'head_id': -1,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool]
                    ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1)]}
        }
        head['device'] = device
        head['dropout_p'] = dropout_p   
        head['criterion'] = criterion
        if 'output_non_linearity' not in head.keys():
            head['output_non_linearity'] = None
        self.num_outputs = head['num_outputs']

        if hasattr(self.backbone,'_fc'):
            fc = self.create_head(head,self.backbone._fc)
            self.header = fc
        else:
            name = model_name.lower()
            meta = models_meta[name]
            modules = list(self.backbone.children())
            l = modules[:meta['head_id']]
            if self.dream_model:
                l+=self.dream_model
            if type(head).__name__ != 'dict':
                backbone = nn.Sequential(*l)
                for layer in head.children():
                    if(type(layer).__name__) == 'StdConv':
                        conv_module = layer
                        break
                conv_layer = conv_module.conv
                temp_args = [conv_layer.out_channels,conv_layer.kernel_size,conv_layer.stride,conv_layer.padding]
                temp_args.insert(0,meta['conv_channels'])
                conv_layer = nn.Conv2d(*temp_args)
                conv_module.conv = conv_layer
                backbone.add_module('custom_head',head)
            else:
                fc = self.create_head(head,modules[-1])
                if adaptive:
                    l += meta['adaptive_head']
                else:
                    l += meta['normal_head']
                if channels == 1:
                    l.insert(0,nn.Conv2d(1,3,3,1,1))
                backbone = nn.Sequential(*l)
                self.header = fc
                # model.add_module('fc',fc)
            self.backbone = backbone
        self.head = head
        
        if type(head).__name__ == 'dict':
            print('Model: {}, Setting head: inputs: {} hidden:{} outputs: {}'.format(model_name,
                                                                          fc.num_inputs,
                                                                          head['layers'],
                                                                          head['num_outputs']))
        else:
            print('Model: {}, Setting head: {}'.format(model_name,type(head).__name__))

    def _get_dropout(self):
        return self.dropout_p
        
    def _set_dropout(self,p=0.45):
        
        if self.model.classifier is not None:
            print('{}: setting head (FC) dropout prob to {:.3f}'.format(self.model_name,p))
            self.model.fc._set_dropout(p=p)

    def get_model_params(self):
        params = super(TransferNetworkImg, self).get_model_params()
        params['class_names'] = self.class_names
        params['num_classes'] = self.num_classes                                              
        params['head'] = self.head
        params['channels'] = self.channels
        return params        

class FacialRecCenterLoss(Network):
    def __init__(self,
                 model_name = 'LeNetPlus',
                 model_type = 'facial_rec',
                 lr = 0.02,
                 lr_feat = 0.1,
                 criterion = nn.CrossEntropyLoss(),
                 optimizer_name_class = 'AdaDelta',
                 optimizer_name_feature = 'AdaDelta',
                 dropout_p = 0.45,
                 add_extra = True,
                 pretrained = True,
                 device = None,
                 best_accuracy = 0.,
                 best_validation_loss=None,
                 best_model_file ='best_facial_rec.pth',
                 class_names = [],
                 num_classes = None,
                 feature_dim = 2,
                 loss_weight = 1,
                 image_size = (256,256)
                 ):

        super().__init__(device=device)
        if model_name.lower() == 'lenetplus':
            self.model = LeNetPlus(num_classes,feature_dim,image_size)
            print('Using LeNet++ for Facial Recognition.')
        else:
            self.set_transfer_model(model_name,pretrained,add_extra)
            self.set_model_head(model_name=model_name,feature_dim=feature_dim,num_classes=num_classes,device=device)
        self.center_loss = CenterLoss(num_classes,feature_dim).to(device)
        self.set_model_params(criterion = criterion,
                              optimizer_name_class = optimizer_name_class,
                              optimizer_name_feature = optimizer_name_feature,
                              lr = lr,
                              lr_feat = lr_feat,
                              dropout_p = dropout_p,
                              model_name = model_name,
                              model_type = model_type,
                              best_accuracy = best_accuracy,
                              best_validation_loss = best_validation_loss,
                              best_model_file = best_model_file,
                              class_names = class_names,
                              num_classes = num_classes,
                              feature_dim = feature_dim,
                              loss_weight = 1
                              )

        self.model = self.model.to(device)
        
    def set_model_params(self,criterion = nn.CrossEntropyLoss(),
                         optimizer_name_class = 'AdaDelta',
                         optimizer_name_feature = 'AdaDelta',
                         lr  = 0.1,
                         lr_feat = 0.1,
                         dropout_p = 0.45,
                         model_name = 'DenseNet',
                         model_type = 'cv_transfer',
                         best_accuracy = 0.,
                         best_validation_loss = None,
                         best_model_file = 'best_model_file.pth',
                         class_names = [],
                         num_classes = None,
                         feature_dim = 2,
                         loss_weight = 1):
                
        super(FacialRecCenterLoss, self).set_model_params(
                                              criterion = criterion,
                                              optimizer_name = optimizer_name_class,
                                              lr = lr,
                                              dropout_p = dropout_p,
                                              model_name = model_name,
                                              model_type = model_type,
                                              best_accuracy = best_accuracy,
                                              best_validation_loss = best_validation_loss,
                                              best_model_file = best_model_file
                                              )
        self.class_names = class_names
        self.num_classes = num_classes
        self.feature_dim = feature_dim                                           
        self.loss_weight = loss_weight
        self.lr_feat = lr_feat
        self.optimizer_feature = get_optim(optimizer_name_feature,self.center_loss.parameters(),lr_feat)
        if len(class_names) == 0:
            self.class_names = {k:str(v) for k,v in enumerate(list(range(self.head['num_outputs'])))}

    def set_transfer_model(self,mname,pretrained=True,add_extra=True,dropout_p = 0.45):   
        self.model = None
        models_dict = {

            'densenet': {'model':models.densenet121(pretrained=pretrained),'conv_channels':1024},
            'resnet34': {'model':models.resnet34(pretrained=pretrained),'conv_channels':512},
            'resnet50': {'model':models.resnet50(pretrained=pretrained),'conv_channels':2048}

        }
        meta = models_dict[mname.lower()]
        try:
            model = meta['model']
            for param in model.parameters():
                param.requires_grad = False
            self.model = model    
            print('Setting transfer learning model: self.model set to {}'.format(mname))
        except:
            print('Setting transfer learning model: model name {} not supported'.format(mname))            

        # creating and adding extra layers to the model
        dream_model = None
        if add_extra:
            channels = meta['conv_channels']
            dream_model = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,1),
                # Printer(),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p),
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p),
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p)
                )        
        self.dream_model = dream_model          
           
    def set_model_head(self,
                        model_name = 'DenseNet',
                        feature_dim = 2,
                        num_classes = 100,
                        adaptive = True,       
                        device = None):

        models_meta = {
        'resnet34': {'conv_channels':512,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'resnet50': {'conv_channels':2048,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'densenet': {'conv_channels':1024,'head_id': -1,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool]
                    ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1)]}
        }

        name = model_name.lower()
        meta = models_meta[name]
        modules = list(self.model.children())
        l = modules[:meta['head_id']]
        if self.dream_model:
            l+=self.dream_model
        fc = modules[-1]
        try:
            in_features =  fc.in_features
        except:
            in_features = fc.model.out.in_features    
        fc1 = nn.Linear(in_features,feature_dim)
        fc2 = nn.Linear(feature_dim,num_classes,bias=False)
        if adaptive:
            l += meta['adaptive_head']
        else:
            l += meta['normal_head']
        l.insert(0,nn.Conv2d(1,3,3,1,1))    
        l.append(Flatten())
        model = nn.Sequential(*l)
        model.add_module('fc1',fc1)
        model.add_module('prelu',nn.PReLU())
        model.add_module('fc2',fc2)
        self.model = model
        
        print('Multi-head set up complete.')

    def forward(self,x):
        if self.model_name.lower() != 'lenetplus':
            l = list(self.model.children())
            for m in l[:-3]:
                x = (m(x))
            features = l[-3](x)        
            classes = l[-1](l[-2](features))
            return features,classes
        else:
            return self.model(x)    

    def compute_loss(self,outputs,labels):
        features, classes = outputs
        # print(self.criterion(classes, labels))
        # print(self.loss_weight * self.center_loss(features, labels))
        loss = self.criterion(classes, labels) + (self.loss_weight * self.center_loss(features,labels))
        return [loss]

    def train_(self,e,trainloader,optimizer,print_every):

        epoch,epochs = e
        self.train()
        t0 = time.time()
        t1 = time.time()
        batches = 0
        running_loss = 0.
        for data_batch in trainloader:
            inputs,labels = data_batch[0],data_batch[1]
            batches += 1
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            self.optimizer_feature.zero_grad()
            outputs = self.forward(inputs)
            loss = self.compute_loss(outputs,labels)[0]
            if self.parallel:
                loss.sum().backward()
                loss = loss.sum()
            else:    
                loss.backward()
                loss = loss.item()
            optimizer.step()
            self.optimizer_feature.step()
            running_loss += loss
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
                        f"Batch training loss: {loss:.3f}\n"
                        f"Average training loss: {running_loss/(batches):.3f}\n"
                      '+----------------------------------------------------------------------+\n'     
                        )
                t0 = time.time()
        return running_loss/len(trainloader)

    def find_lr(self,trn_loader,init_value=1e-8,final_value=10.,beta=0.98,plot=False):
        
        print('\nFinding the ideal learning rates.')

        num = len(trn_loader)-1
        mult = (final_value / init_value) ** (1/num)        
        model_state = copy.deepcopy(self.model.state_dict())
        optim_state = copy.deepcopy(self.optimizer.state_dict())
        optim_state2 = copy.deepcopy(self.optimizer_feature.state_dict())
        optimizer = self.optimizer
        optimizer2 = self.optimizer_feature
        lr = init_value
        lr2 = init_value
        optimizer.param_groups[0]['lr'] = lr
        optimizer2.param_groups[0]['lr'] = lr2
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        log_lrs2 = []
        for data_batch in trn_loader:
            batch_num += 1
            inputs,labels = data_batch[0],data_batch[1]
            inputs = inputs.to(self.device)           
            labels = labels.to(self.device)
            optimizer.zero_grad()
            optimizer2.zero_grad()
            outputs = self.forward(inputs)
            loss = self.compute_loss(outputs,labels)[0]
            #Compute the smoothed loss
            if self.parallel:   
                avg_loss = beta * avg_loss + (1-beta) * loss.sum()    
            else:    
                avg_loss = beta * avg_loss + (1-beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                self.log_lrs,self.log_lrs2, self.find_lr_losses = log_lrs,log_lrs2,losses
                self.model.load_state_dict(model_state)
                self.optimizer.load_state_dict(optim_state)
                self.optimizer_feature.load_state_dict(optim_state2)
                if plot:
                    self.plot_find_lr()
                temp_lr = self.log_lrs[np.argmin(self.find_lr_losses)-(len(self.log_lrs)//8)]
                temp_lr2 = self.log_lrs2[np.argmin(self.find_lr_losses)-(len(self.log_lrs2)//8)]
                self.lr = (10**temp_lr)
                self.lr_feat = (10**temp_lr2)
                print('Found them: {} and {}\n'.format(self.lr,self.lr_feat))
                return self.lr,self.lr_feat
            #Record the best loss
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            log_lrs2.append(math.log10(lr2))
            #Do the SGD step
            if self.parallel:
                loss.sum().backward()
            else:    
                loss.backward()
            optimizer.step()
            optimizer2.step()
            #Update the lr for the next step
            lr *= mult
            lr2 *= mult
            optimizer.param_groups[0]['lr'] = lr
            optimizer2.param_groups[0]['lr'] = lr2

        self.log_lrs,self.log_lrs2, self.find_lr_losses = log_lrs,log_lrs2,losses
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optim_state)
        self.optimizer_feature.load_state_dict(optim_state2)
        if plot:
            self.plot_find_lr()
        temp_lr = self.log_lrs[np.argmin(self.find_lr_losses)-(len(self.log_lrs)//10)]
        temp_lr2 = self.log_lrs2[np.argmin(self.find_lr_losses)-(len(self.log_lrs2)//10)]
        self.lr = (10**temp_lr)
        self.lr_feat = (10**temp_lr2)
        print('Found them: {} and {}\n'.format(self.lr,self.lr_feat))
        return self.lr,self.lr_feat

class EmotionRec(FC):
    def __init__(self,
                num_outputs=10,
                layers=[],
                lr=0.003,
                class_names=[],
                extractor_name = 'resnet18',
                extractor_channels = 3,
                optimizer_name='AdaDelta',
                dropout_p=0.2,
                hidden_non_linearity='relu',
                output_non_linearity=None,
                criterion=nn.CrossEntropyLoss(),
                model_name='Emotion Classifier',
                model_type ='classifier',
                best_accuracy=0.,
                best_validation_loss=None,
                best_model_file = 'best_lm_model_file.pth',
                device=None):
        extractor_name = extractor_name.lower()
        models_dict = {

            'resnet18': {'model':models.resnet18(pretrained=True),'output': 512},
            'resnet34': {'model':models.resnet34(pretrained=True),'output':512},
            'resnet50': {'model':models.resnet50(pretrained=True),'output':2048}

        }
        try:
            meta = models_dict[extractor_name]
            children = list(meta['model'].children())[:-1]
            if extractor_channels == 1:
                children.insert(0,nn.Conv2d(1,3,3,1,1))
            model = nn.Sequential(*children)
            model.add_module('flatten',Flatten())
            for param in model.parameters():
                param.requires_grad = False
            num_inputs = meta['output']+144
            super().__init__(num_inputs=num_inputs,
                num_outputs=num_outputs,
                layers=layers,
                lr=lr,
                class_names=class_names,
                optimizer_name=optimizer_name,
                dropout_p=dropout_p,
                hidden_non_linearity=hidden_non_linearity,
                output_non_linearity=output_non_linearity,
                criterion=criterion,
                model_name=model_name,
                model_type=model_type,
                best_accuracy=best_accuracy,
                best_validation_loss=best_validation_loss,
                best_model_file = best_model_file,
                device=device)
            self.extractor = model.to(device)   
            print('Feature Extractor set to {}'.format(extractor_name))
            print('{} setup complete.'.format(model_name))
        except:
            print('Feature Extractor {} not supported'.format(extractor_name))
    
    def train_(self,e,trainloader,optimizer,print_every):

        epoch,epochs = e
        self.train()
        t0 = time.time()
        t1 = time.time()
        batches = 0
        running_loss = 0.
        for data_batch in trainloader:
            inputs,labels,landmarks = data_batch[0],data_batch[1],data_batch[2]
            batches += 1
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            landmarks = landmarks.to(self.device)
            features = self.extractor(inputs)
            model_inputs = torch.cat([features,landmarks],dim=1)
            optimizer.zero_grad()
            outputs = self.forward(model_inputs)
            loss = self.compute_loss(outputs,labels)[0]
            if self.parallel:
                loss.sum().backward()
                loss = loss.sum()
            else:    
                loss.backward()
                loss = loss.item()
            optimizer.step()
            running_loss += loss
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
                        f"Batch training loss: {loss:.3f}\n"
                        f"Average training loss: {running_loss/(batches):.3f}\n"
                      '+----------------------------------------------------------------------+\n'     
                        )
                t0 = time.time()
        return running_loss/len(trainloader)
    
    def evaluate(self,dataloader,metric='accuracy'):
        
        running_loss = 0.
        classifier = None

        if self.model_type == 'classifier':# or self.num_classes is not None:
           classifier = Classifier(self.class_names)

        y_pred = []
        y_true = []

        self.eval()
        rmse_ = 0.
        with torch.no_grad():
            for data_batch in dataloader:
                inputs,labels,landmarks = data_batch[0],data_batch[1],data_batch[2]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                landmarks = landmarks.to(self.device)
                features = self.extractor(inputs)
                model_inputs = torch.cat([features,landmarks],dim=1)
                outputs = self.forward(model_inputs)
                loss = self.compute_loss(outputs,labels)[0]
                if self.parallel:
                    running_loss += loss.sum()
                    outputs = parallel.gather(outputs,self.device)
                else:        
                    running_loss += loss.item()
                if classifier is not None and metric == 'accuracy':
                    classifier.update_accuracies(outputs,labels)
                    y_true.extend(list(labels.squeeze(0).cpu().numpy()))
                    _, preds = torch.max(torch.exp(outputs), 1)
                    y_pred.extend(list(preds.cpu().numpy()))
                elif metric == 'rmse':
                    rmse_ += rmse(outputs,labels).cpu().numpy()
            
        self.train()

        ret = {}
        # print('Running_loss: {:.3f}'.format(running_loss))
        if metric == 'rmse':
            print('Total rmse: {:.3f}'.format(rmse_))
            ret['final_rmse'] = rmse_/len(dataloader)

        ret['final_loss'] = running_loss/len(dataloader)

        if classifier is not None:
            ret['accuracy'],ret['class_accuracies'] = classifier.get_final_accuracies()
            ret['report'] = classification_report(y_true,y_pred,target_names=self.class_names)
            ret['confusion_matrix'] = confusion_matrix(y_true,y_pred)
            try:
                ret['roc_auc_score'] = roc_auc_score(y_true,y_pred)
            except:
                pass
        return ret

    def find_lr(self,trn_loader,init_value=1e-8,final_value=10.,beta=0.98,plot=False):
        
        print('\nFinding the ideal learning rate.')

        model_state = copy.deepcopy(self.model.state_dict())
        optim_state = copy.deepcopy(self.optimizer.state_dict())
        optimizer = self.optimizer
        num = len(trn_loader)-1
        mult = (final_value / init_value) ** (1/num)
        lr = init_value
        optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for data_batch in trn_loader:
            batch_num += 1
            inputs,labels,landmarks = data_batch[0],data_batch[1],data_batch[2]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            landmarks = landmarks.to(self.device)
            features = self.extractor(inputs)
            model_inputs = torch.cat([features,landmarks],dim=1)
            optimizer.zero_grad()
            outputs = self.forward(model_inputs)
            loss = self.compute_loss(outputs,labels)[0]
            #Compute the smoothed loss
            if self.parallel:   
                avg_loss = beta * avg_loss + (1-beta) * loss.sum()    
            else:    
                avg_loss = beta * avg_loss + (1-beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                self.log_lrs, self.find_lr_losses = log_lrs,losses
                self.model.load_state_dict(model_state)
                self.optimizer.load_state_dict(optim_state)
                if plot:
                    self.plot_find_lr()
                temp_lr = self.log_lrs[np.argmin(self.find_lr_losses)-(len(self.log_lrs)//8)]
                self.lr = (10**temp_lr)
                print('Found it: {}\n'.format(self.lr))
                return self.lr
            #Record the best loss
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            #Do the SGD step
            if self.parallel:
                loss.sum().backward()
            else:    
                loss.backward()
            optimizer.step()
            #Update the lr for the next step
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr

        self.log_lrs, self.find_lr_losses = log_lrs,losses
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optim_state)
        if plot:
            self.plot_find_lr()
        temp_lr = self.log_lrs[np.argmin(self.find_lr_losses)-(len(self.log_lrs)//10)]
        self.lr = (10**temp_lr)
        print('Found it: {}\n'.format(self.lr))
        return self.lr

    def process_input(self,imgs = [],size = (500,500),norm = False):
        if size is not None:
            imgs = [cv2.resize(img,size) for img in imgs]
        else:
            size = imgs[0].shape[:2]
        landmarks = [face_recognition.face_landmarks(img) for img in imgs]
        landmarks = np.array([process_landmarks(lm) for lm in landmarks])
        marks_idx = landmarks[:,1].astype(np.bool)
        landmarks = (torch.Tensor(list(landmarks[:,0][marks_idx]))-size[0])/size[0]
        imgs = np.array(imgs)[marks_idx]
        if len(imgs) == 0:
            return None,[]
        imgs_g = [rgb2gray(i)[...,None] for i in imgs]
        batch = imgs_to_batch(imgs = imgs_g,size = None,norm = norm,show = False).to(self.device)
        landmarks = landmarks.to(self.device)
        features = self.extractor(batch)
        model_inputs = torch.cat([features,landmarks],dim=1)
        return model_inputs,imgs

    def classify(self,imgs = [],size = (512,512),norm = False,class_names = None):
        inputs,imgs = self.process_input(imgs,size,norm)
        if len(imgs) == 0:
            class_preds = ['']
        else:    
            if class_names is None:
                class_names = self.class_names
            outputs = self.predict(inputs)
            try:    
                _, preds = torch.max(torch.exp(outputs), 1)
                # _, preds = torch.max(outputs, 1)
            except:
                _, preds = torch.max(torch.exp(outputs.unsqueeze(0)), 1)
            class_preds = [str(class_names[p]) for p in preds]
        return class_preds,imgs

class SemanticSegmentation(Network):

    def __init__(self,
                model_name='HRNet',
                model_type='segmentation',
                config = {
                    'STAGE1':{'NUM_MODULES':1,'NUM_BRANCHES':1,'NUM_BLOCKS':[2],'NUM_CHANNELS':[64],
                                'BLOCK':'BOTTLENECK','FUSE_METHOD':'SUM'},
                    'STAGE2':{'NUM_MODULES':1,'NUM_BRANCHES':2,'NUM_BLOCKS':[2,2],'NUM_CHANNELS':[48,96],
                                'BLOCK':'BASIC','FUSE_METHOD':'SUM'},
                    'STAGE3':{'NUM_MODULES':4,'NUM_BRANCHES':3,'NUM_BLOCKS':[2,2,2],'NUM_CHANNELS':[48,96,192],
                                'BLOCK':'BASIC','FUSE_METHOD':'SUM'},
                    'STAGE4':{'NUM_MODULES':3,'NUM_BRANCHES':4,'NUM_BLOCKS':[2,2,2,2],'NUM_CHANNELS':[48,96,192,384],
                                'BLOCK':'BASIC','FUSE_METHOD':'SUM'},
                    'FINAL_CONV_KERNEL':1,
                    'NUM_CLASSES':10
                    },
                lr=0.02,
                optimizer_name = 'AdaDelta',
                criterion = nn.CrossEntropyLoss(),
                dropout_p=0.45,
                device='cpu',
                best_accuracy=0.,
                best_validation_loss=None,
                best_model_file ='best_seg_model.pth',
                channels = 3,
                class_names = []
                ):

        super().__init__(device=device)

        self.model = get_seg_model(config)
        num_classes = config['NUM_CLASSES']
        self.set_model_params(
                            config = config,
                            criterion=criterion,
                            optimizer_name = optimizer_name,
                            lr = lr,
                            dropout_p = dropout_p,
                            model_name = model_name,
                            model_type = model_type,
                            best_accuracy = best_accuracy,
                            best_validation_loss = best_validation_loss,
                            best_model_file = best_model_file,
                            class_names = class_names,
                            num_classes = num_classes,
                            channels = channels
                            )

        self.model = self.model.to(device)
        
    def set_model_params(self,
                            config = {
                                'STAGE1':{'NUM_MODULES':1,'NUM_BRANCHES':1,'NUM_BLOCKS':[2],'NUM_CHANNELS':[64],
                                            'BLOCK':'BOTTLENECK','FUSE_METHOD':'SUM'},
                                'STAGE2':{'NUM_MODULES':1,'NUM_BRANCHES':2,'NUM_BLOCKS':[2,2],'NUM_CHANNELS':[48,96],
                                            'BLOCK':'BASIC','FUSE_METHOD':'SUM'},
                                'STAGE3':{'NUM_MODULES':4,'NUM_BRANCHES':3,'NUM_BLOCKS':[2,2,2],'NUM_CHANNELS':[48,96,192],
                                            'BLOCK':'BASIC','FUSE_METHOD':'SUM'},
                                'STAGE4':{'NUM_MODULES':3,'NUM_BRANCHES':4,'NUM_BLOCKS':[2,2,2,2],'NUM_CHANNELS':[48,96,192,384],
                                            'BLOCK':'BASIC','FUSE_METHOD':'SUM'},
                                'FINAL_CONV_KERNEL':1,
                                'NUM_CLASSES':10
                                },
                            criterion = nn.CrossEntropyLoss(),
                            optimizer_name = 'AdaDelta',
                            lr  = 0.1,
                            dropout_p = 0.45,
                            model_name = 'HRNet',
                            model_type = 'segmentation',
                            best_accuracy = 0.,
                            best_validation_loss = None,
                            best_model_file = 'best_seg_model.pth',
                            class_names = [],
                            num_classes = None,
                            channels = 3):
        print('Using HRNet for Semantic Segmentation.')
        super(SemanticSegmentation, self).set_model_params(
                                                criterion = criterion,
                                                optimizer_name = optimizer_name,
                                                lr = lr,
                                                dropout_p = dropout_p,
                                                model_name = model_name,
                                                model_type = model_type,
                                                best_accuracy = best_accuracy,
                                                best_validation_loss = best_validation_loss,
                                                best_model_file = best_model_file
                                                )
        if best_validation_loss is not None:
            print('Current best loss = {:.3f}'.format(best_validation_loss))
        self.class_names = class_names
        self.num_classes = num_classes                                              
        self.channels = channels    

    def get_model_params(self):
        params = super(SemanticSegmentation, self).get_model_params()
        params['class_names'] = self.class_names
        params['num_classes'] = self.num_classes
        return params

class TransferSemanticSegmentation(Network):

    def __init__(self,
                model_name='DeepLabV3_ResNet50',
                model_type='segmentation',
                config = {
                    'STAGE1':{'NUM_MODULES':1,'NUM_BRANCHES':1,'NUM_BLOCKS':[2],'NUM_CHANNELS':[64],
                                'BLOCK':'BOTTLENECK','FUSE_METHOD':'SUM'},
                    'STAGE2':{'NUM_MODULES':1,'NUM_BRANCHES':2,'NUM_BLOCKS':[2,2],'NUM_CHANNELS':[48,96],
                                'BLOCK':'BASIC','FUSE_METHOD':'SUM'},
                    'STAGE3':{'NUM_MODULES':4,'NUM_BRANCHES':3,'NUM_BLOCKS':[2,2,2],'NUM_CHANNELS':[48,96,192],
                                'BLOCK':'BASIC','FUSE_METHOD':'SUM'},
                    'STAGE4':{'NUM_MODULES':3,'NUM_BRANCHES':4,'NUM_BLOCKS':[2,2,2,2],'NUM_CHANNELS':[48,96,192,384],
                                'BLOCK':'BASIC','FUSE_METHOD':'SUM'},
                    'FINAL_CONV_KERNEL':1,
                    'NUM_CLASSES':10
                    },
                lr=0.02,
                optimizer_name = 'AdaDelta',
                criterion = nn.CrossEntropyLoss(),
                dropout_p = 0.45,
                device='cpu',
                best_accuracy = 0.,
                best_validation_loss=None,
                best_model_file = 'best_seg_model.pth',
                channels = 3,
                add_extra = True,
                pretrained = True,
                class_names = [],
                num_classes = None
                ):

        super().__init__(device=device)
        if num_classes is None:
            num_classes = config['NUM_CLASSES']
        self.set_transfer_model(model_name,num_classes,add_extra,dropout_p,pretrained)
        self.set_model_params(
                            config = config,
                            criterion=criterion,
                            optimizer_name = optimizer_name,
                            lr = lr,
                            dropout_p = dropout_p,
                            model_name = model_name,
                            model_type = model_type,
                            best_accuracy = best_accuracy,
                            best_validation_loss = best_validation_loss,
                            best_model_file = best_model_file,
                            class_names = class_names,
                            num_classes = num_classes,
                            channels = channels
                            )

        self.model = self.model.to(device)

    def compute_loss(self,outputs,labels):
        seg_mask = outputs['out']
        aux = outputs['aux']
        loss1 = self.criterion(seg_mask,labels)
        loss2 = self.criterion(aux,labels)
        # print(loss1,loss2)
        return [loss1+(0.2*loss2)]

    def set_transfer_model(self,model_name,num_classes,add_extra,dropout_p,pretrained):
        models_dict = {'deeplabv3_resnet50':deeplabv3_resnet50,'deeplabv3_resnet101':deeplabv3_resnet101}
        model_name = model_name.lower()
        model = models_dict[model_name](pretrained=pretrained,aux_loss=True)
        for p in model.parameters():
            p.requires_grad = False
        classifier = list(model.classifier.children())
        last_conv = classifier[-1]
        params = [last_conv.in_channels,last_conv.out_channels,last_conv.kernel_size,last_conv.stride,last_conv.padding]
        params[1] = num_classes
        if add_extra:
            channels = params[0]
            dream_model = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,1),
                # Printer(),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p),
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p)
                # nn.Conv2d(channels,channels,3,1,1),
                # nn.BatchNorm2d(channels),
                # nn.ReLU(True),
                # nn.Dropout2d(dropout_p)
                )
        classifier.insert(-1,dream_model)
        classifier[-1] = nn.Conv2d(*params)
        c = nn.Sequential(*classifier)
        for p in c.parameters():
            p.requires_grad = True
        model.add_module('classifier',c)
        self.model = model

    def set_model_params(self,
                            config = {
                                'STAGE1':{'NUM_MODULES':1,'NUM_BRANCHES':1,'NUM_BLOCKS':[2],'NUM_CHANNELS':[64],
                                            'BLOCK':'BOTTLENECK','FUSE_METHOD':'SUM'},
                                'STAGE2':{'NUM_MODULES':1,'NUM_BRANCHES':2,'NUM_BLOCKS':[2,2],'NUM_CHANNELS':[48,96],
                                            'BLOCK':'BASIC','FUSE_METHOD':'SUM'},
                                'STAGE3':{'NUM_MODULES':4,'NUM_BRANCHES':3,'NUM_BLOCKS':[2,2,2],'NUM_CHANNELS':[48,96,192],
                                            'BLOCK':'BASIC','FUSE_METHOD':'SUM'},
                                'STAGE4':{'NUM_MODULES':3,'NUM_BRANCHES':4,'NUM_BLOCKS':[2,2,2,2],'NUM_CHANNELS':[48,96,192,384],
                                            'BLOCK':'BASIC','FUSE_METHOD':'SUM'},
                                'FINAL_CONV_KERNEL':1,
                                'NUM_CLASSES':10
                                },
                            criterion = nn.CrossEntropyLoss(),
                            optimizer_name = 'AdaDelta',
                            lr  = 0.1,
                            dropout_p = 0.45,
                            model_name = 'HRNet',
                            model_type = 'segmentation',
                            best_accuracy = 0.,
                            best_validation_loss = None,
                            best_model_file = 'best_seg_model.pth',
                            class_names = [],
                            num_classes = None,
                            channels = 3):
        print('Using {} for Semantic Segmentation.'.format(model_name))
        super(TransferSemanticSegmentation, self).set_model_params(
                                                criterion = criterion,
                                                optimizer_name = optimizer_name,
                                                lr = lr,
                                                dropout_p = dropout_p,
                                                model_name = model_name,
                                                model_type = model_type,
                                                best_accuracy = best_accuracy,
                                                best_validation_loss = best_validation_loss,
                                                best_model_file = best_model_file
                                                )
        if best_validation_loss is not None:
            print('Current best loss = {:.3f}'.format(best_validation_loss))
        self.class_names = class_names
        self.num_classes = num_classes                                              
        self.channels = channels    

    def fit(self,trainloader,validloader,epochs=2,print_every=10,validate_every=1,save_best_every=1):

        optim_path = Path(self.best_model_file)
        optim_path = optim_path.stem + '_optim' + optim_path.suffix
        with mlflow.start_run() as run:
            for epoch in range(epochs):
                self.model = self.model.to(self.device)
                mlflow.log_param('epochs',epochs)
                mlflow.log_param('lr',self.optimizer.param_groups[0]['lr'])
                mlflow.log_param('bs',trainloader.batch_size)
                print('Epoch:{:3d}/{}\n'.format(epoch+1,epochs))
                epoch_train_loss =  self.train_((epoch,epochs),trainloader,self.optimizer,print_every)  
                        
                if  validate_every and (epoch % validate_every == 0):
                    t2 = time.time()
                    eval_dict = self.evaluate(validloader)
                    epoch_validation_loss = eval_dict['final_loss']
                    epoch_validation_dice = eval_dict['dice_score']
                    if self.parallel:
                        try:
                            epoch_train_loss = epoch_train_loss.item()
                            epoch_validation_loss = epoch_validation_loss.item()
                        except:
                            pass  
                    mlflow.log_metric('Train Loss',epoch_train_loss)
                    mlflow.log_metric('Validation Loss',epoch_validation_loss)
                    mlflow.log_metric('Dice Score',epoch_validation_dice)
                    
                    time_elapsed = time.time() - t2
                    if time_elapsed > 60:
                        time_elapsed /= 60.
                        measure = 'min'
                    else:
                        measure = 'sec'    
                    print('\n'+'/'*36+'\n'
                            f"{time.asctime().split()[-2]}\n"
                            f"Epoch {epoch+1}/{epochs}\n"    
                            f"Validation time: {time_elapsed:.6f} {measure}\n"    
                            f"Epoch training loss: {epoch_train_loss:.6f}\n"                        
                            f"Epoch validation loss: {epoch_validation_loss:.6f}\n"
                            f"Epoch validation dice coefficient: {epoch_validation_dice:.6f}"
                        )
                    if self.model_type == 'classifier':# or self.num_classes is not None:
                        epoch_accuracy = eval_dict['accuracy']
                        mlflow.log_metric('Validation Accuracy',epoch_accuracy)
                        print("Validation accuracy: {:.3f}".format(epoch_accuracy))
                        # print('\\'*36+'/'*36+'\n')
                        print('\\'*36+'\n')
                        if self.best_accuracy == 0. or (epoch_accuracy >= self.best_accuracy):
                            print('\n**********Updating best accuracy**********\n')
                            print('Previous best: {:.3f}'.format(self.best_accuracy))
                            print('New best: {:.3f}\n'.format(epoch_accuracy))
                            print('******************************************\n')
                            self.best_accuracy = epoch_accuracy
                            mlflow.log_metric('Best Accuracy',self.best_accuracy)
                            optim_path = Path(self.best_model_file)
                            optim_path = optim_path.stem + '_optim' + optim_path.suffix
                            torch.save(self.model.state_dict(),self.best_model_file)
                            torch.save(self.optimizer.state_dict(),optim_path)     
                            mlflow.pytorch.log_model(self,'mlflow_logged_models')
                            curr_time = str(datetime.now())
                            curr_time = '_'+curr_time.split()[1].split('.')[0]
                            mlflow_save_path = Path('mlflow_saved_training_models')/\
                                               (Path(self.best_model_file).stem+'_{}_{}'.format(str(round(epoch_accuracy,2)),str(epoch)+curr_time))
                            mlflow.pytorch.save_model(self,mlflow_save_path)
                    else:
                        print('\\'*36+'\n')
                        if self.best_validation_loss == None or (epoch_validation_loss <= self.best_validation_loss):
                            print('\n**********Updating best validation loss**********\n')
                            if self.best_validation_loss is not None:
                                print('Previous best: {:.7f}'.format(self.best_validation_loss))
                            print('New best loss = {:.7f}\n'.format(epoch_validation_loss))
                            print('*'*49+'\n')
                            self.best_validation_loss = epoch_validation_loss
                            mlflow.log_metric('Best Loss',self.best_validation_loss)
                            optim_path = Path(self.best_model_file)
                            optim_path = optim_path.stem + '_optim' + optim_path.suffix
                            torch.save(self.model.state_dict(),self.best_model_file)
                            torch.save(self.optimizer.state_dict(),optim_path)     
                            mlflow.pytorch.log_model(self,'mlflow_logged_models')
                            curr_time = str(datetime.now())
                            curr_time = '_'+curr_time.split()[1].split('.')[0]
                            mlflow_save_path = Path('mlflow_saved_training_models')/\
                                (Path(self.best_model_file).stem+'_{}_{}'.format(str(round(epoch_validation_loss,3)),str(epoch)+curr_time))
                            mlflow.pytorch.save_model(self,mlflow_save_path)
                        
                    self.train()
        torch.cuda.empty_cache()
        try:
            print('\nLoaded best model\n')
            self.model.load_state_dict(torch.load(self.best_model_file))
            self.optimizer.load_state_dict(torch.load(optim_path))
            os.remove(self.best_model_file)
            os.remove(optim_path)
        except:
            pass    

    def evaluate(self,dataloader):
        
        running_loss = 0.
        running_dice = 0.
        self.eval()
        with torch.no_grad():
            for data_batch in dataloader:
                inputs, labels = data_batch[0],data_batch[1]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(inputs)
                loss = self.compute_loss(outputs,labels)[0]
                # dice_score = dice(outputs['out'],labels,iou=False)
                dice_score = dice_coeff(outputs['out'],labels)
                # print(dice_score.item())
                if self.parallel:
                    running_loss += loss.sum()
                    outputs = parallel.gather(outputs,self.device)
                else:        
                    running_loss += loss.item()
                    running_dice += dice_score.item()
            
        self.train()
        ret = {}
        ret['final_loss'] = running_loss/len(dataloader)
        ret['dice_score'] = running_dice/len(dataloader)
        return ret

    def get_model_params(self):
        params = super(TransferSemanticSegmentation, self).get_model_params()
        params['class_names'] = self.class_names
        params['num_classes'] = self.num_classes
        return params

class ObjectDetection(Network):

    def __init__(self,
                model_name='resnet34',
                model_type='object_detection',
                lr=0.02,
                optimizer_name = 'AdaDelta',
                dropout_p = 0.45,
                device = 'cpu',
                best_accuracy = 0.,
                best_validation_loss = None,
                best_model_file = 'best_obj_model.pth',
                channels = 3,
                class_names = [],
                num_classes = None
                ):

        super().__init__(device=device)

        self.set_model(model_name,num_classes,device)
        self.criterion = SSD300.MultiBoxLoss(priors_cxcy=self.model.priors_cxcy,device=device).to(device)
        self.set_model_params(
                            optimizer_name = optimizer_name,
                            lr = lr,
                            dropout_p = dropout_p,
                            model_name = model_name,
                            model_type = model_type,
                            best_accuracy = best_accuracy,
                            best_validation_loss = best_validation_loss,
                            best_model_file = best_model_file,
                            class_names = class_names,
                            num_classes = num_classes,
                            channels = channels
                            )
        self.model = self.model.to(device)

    def set_model(self,model_name,num_classes,device):
        models_dict = {'resnet34': {'model':resnet34, 'out_channels': [128,256]}, 'resnet50':{'model':resnet50, 'out_channels':[512,1024]}}
        model_base = SSD.ResnetBase(models_dict[model_name.lower()]).to(device)
        self.model = SSD.SSD300(model_base,num_classes,device)
        
    def set_model_params(self,
                            optimizer_name = 'AdaDelta',
                            lr  = 0.1,
                            dropout_p = 0.45,
                            model_name = 'resnet34',
                            model_type = 'object_detection',
                            best_accuracy = 0.,
                            best_validation_loss = None,
                            best_model_file = 'best_obj_model.pth',
                            class_names = [],
                            num_classes = None,
                            channels = 3):
        print('Using SSD300 with base {} for Object Detection.'.format(model_name))
        if best_validation_loss is not None:
            print('Current best loss = {:.3f}'.format(best_validation_loss))
        self.optimizer_name = optimizer_name
        self.set_optimizer(self.model.parameters(),optimizer_name,lr=lr)
        self.lr = lr
        self.dropout_p = dropout_p
        self.model_name =  model_name
        self.model_type = model_type
        self.best_accuracy = best_accuracy
        self.best_validation_loss = best_validation_loss
        self.best_model_file = best_model_file
        self.class_names = class_names
        self.num_classes = num_classes                                              
        self.channels = channels    

    def get_model_params(self):
        params = super(ObjectDetection, self).get_model_params()
        params['class_names'] = self.class_names
        params['num_classes'] = self.num_classes
        return params
    
    def compute_loss(self,predicted_locs,predicted_scores,boxes,labels):
        return [self.criterion(predicted_locs,predicted_scores,boxes,labels)]

    def batch_to_loss(self,data_batch):
        inputs,boxes,labels = data_batch[0],data_batch[1],data_batch[2]
        inputs = inputs.to(self.device)           
        boxes = [b.to(self.device) for b in boxes]
        labels = [l.to(self.device) for l in labels]
        predicted_locs, predicted_scores = self.forward(inputs)
        loss = self.compute_loss(predicted_locs,predicted_scores,boxes,labels)[0]
        return loss
    
    def evaluate_obj(self,test_loader):
        """
        Evaluate.

        :param test_loader: DataLoader for test data
        :param model: model
        """
        pp = PrettyPrinter()
        # Make sure it's in eval mode
        self.model.eval()

        # Lists to store detected and true boxes, labels, scores
        det_boxes = list()
        det_labels = list()
        det_scores = list()
        true_boxes = list()
        true_labels = list()
        true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

        with torch.no_grad():
            # Batches
            for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
                images = images.to(self.device)  # (N, 3, 300, 300)

                # Forward prop.
                predicted_locs, predicted_scores = self.model(images)

                # Detect objects in SSD output
                det_boxes_batch, det_labels_batch, det_scores_batch = self.model.detect_objects(predicted_locs, predicted_scores,
                                                                                        min_score=0.01, max_overlap=0.45,
                                                                                        top_k=200)
                # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results

                # Store this batch's results for mAP calculation
                boxes = [b.to(self.device) for b in boxes]
                labels = [l.to(self.device) for l in labels]
                difficulties = [d.to(self.device) for d in difficulties]

                det_boxes.extend(det_boxes_batch)
                det_labels.extend(det_labels_batch)
                det_scores.extend(det_scores_batch)
                true_boxes.extend(boxes)
                true_labels.extend(labels)
                true_difficulties.extend(difficulties)

            # Calculate mAP
            APs, mAP = obj_utils.calculate_mAP(det_boxes, det_labels, det_scores,
                    true_boxes, true_labels, true_difficulties,self.class_names,self.device)

        # Print AP for each class
        pp.pprint(APs)

        print('\nMean Average Precision (mAP): %.3f' % mAP)

    def detect(self,img,label_color_map, min_score=0.2, max_overlap=0.5, top_k=200, suppress=None):
        """
        Detect objects in an image with a trained SSD300, and visualize the results.

        :param original_image: image, a PIL Image
        :param min_score: minimum threshold for a detected box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
        :return: annotated image, a PIL Image
        """

        rev_label_map = {v: k for k, v in self.class_names.items()}

        # Transform
        original_image = Image.fromarray(img)
        image = imgs_to_batch(imgs = [img],size = (300,300),norm = True, device = self.device)

        # Forward prop.
        predicted_locs, predicted_scores = self.model(image)

        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = self.model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                                max_overlap=max_overlap, top_k=top_k)

        # Move detections to the CPU
        det_boxes = det_boxes[0].to('cpu')

        # Transform to original image dimensions
        original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
        det_boxes = det_boxes * original_dims

        # Decode class integer labels
        det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

        # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
        if det_labels == ['background']:
            # Just return original image
            return np.array(original_image)

        # Annotate
        annotated_image = original_image
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.truetype("/home/farhan/hamza/dreamai_mlflow/Calibri.ttf", 15)

        # Suppress specific classes, if needed
        for i in range(det_boxes.size(0)):
            if suppress is not None:
                if det_labels[i] in suppress:
                    continue

            # Boxes
            box_location = det_boxes[i].tolist()
            draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
            draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
                det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
            # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
            #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
            # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
            #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

            # Text
            text_size = font.getsize(det_labels[i].upper())
            text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
            textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                                box_location[1]]
            draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
            draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                    font=font)
        del draw

        return np.array(annotated_image)
