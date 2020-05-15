from dreamai.utils import *
from dreamai import parallel
from dreamai.dai_imports import *
from dreamai.parallel import DataParallelModel, DataParallelCriterion

class Classifier():
    def __init__(self,class_names):
        self.class_names = class_names
        self.class_correct = defaultdict(int)
        self.class_totals = defaultdict(int)

    def update_accuracies(self,outputs,labels):
        _, preds = torch.max(torch.exp(outputs), 1)
        # _, preds = torch.max(outputs, 1)
        correct = np.squeeze(preds.eq(labels.data.view_as(preds)))
        for i in range(labels.shape[0]):
            label = labels.data[i].item()
            self.class_correct[label] += correct[i].item()
            self.class_totals[label] += 1

    def fai_update_multi_accuracies(self, preds, label):
        correct = label*preds
        class_idx = torch.nonzero(label)[0]
        for idx in class_idx:
            c = correct[idx].item()
            idx = idx.item()
            self.class_correct[idx] += c
            self.class_totals[idx] += 1

    def update_multi_accuracies(self,outputs,labels,thresh=0.5):
        preds = torch.sigmoid(outputs) > thresh
        correct = (labels==1)*(preds==1)
        for i in range(labels.shape[0]):
            label = torch.nonzero(labels.data[i]).squeeze(1)
            for l in label:
                c = correct[i][l].item()
                l = l.item()
                self.class_correct[l] += c
                self.class_totals[l] += 1
    
    def update_tta_accuracies(self,preds,labels):
        # _, preds = torch.max(torch.exp(outputs), 1)
        # _, preds = torch.max(outputs, 1)
        correct = np.squeeze(preds.eq(labels.data.view_as(preds)))
        for i in range(labels.shape[0]):
            label = labels.data[i].item()
            self.class_correct[label] += correct[i].item()
            self.class_totals[label] += 1

    def update_multi_tta_accuracies(self,preds,labels):
        # preds = torch.sigmoid(outputs) > thresh
        correct = (labels==1)*(preds==1)
        for i in range(labels.shape[0]):
            label = torch.nonzero(labels.data[i]).squeeze(1)
            for l in label:
                c = correct[i][l].item()
                l = l.item()
                self.class_correct[l] += c
                self.class_totals[l] += 1

    def get_final_accuracies(self):
        accuracy = (100*np.sum(list(self.class_correct.values()))/np.sum(list(self.class_totals.values())))
        try:
            class_accuracies = [(self.class_names[i],100.0*(self.class_correct[i]/self.class_totals[i])) 
                                 for i in self.class_names.keys() if self.class_totals[i] > 0]
        except:
            class_accuracies = [(self.class_names[i],100.0*(self.class_correct[i]/self.class_totals[i])) 
                                 for i in range(len(self.class_names)) if self.class_totals[i] > 0]
        return accuracy, class_accuracies

# class MultiLabelClassifier():
#     def __init__(self,class_names):
#         self.class_names = class_names
#         self.class_correct = defaultdict(int)
#         self.class_totals = defaultdict(int)

#     def update_accuracies(self,outputs,labels,thresh=0.5):
#         preds = torch.sigmoid(outputs) > thresh
#         correct = (labels==1)*(preds==1)
#         for i in range(labels.shape[0]):
#             label = torch.nonzero(labels.data[i]).squeeze(1)
#             for l in label:
#                 c = correct[i][l].item()
#                 l = l.item()
#                 self.class_correct[l] += c
#                 self.class_totals[l] += 1

#     def get_final_accuracies(self):
#         accuracy = (100*np.sum(list(self.class_correct.values()))/np.sum(list(self.class_totals.values())))
#         try:
#             class_accuracies = [(self.class_names[i],100.0*(self.class_correct[i]/self.class_totals[i])) 
#                                  for i in self.class_names.keys() if self.class_totals[i] > 0]
#         except:
#             class_accuracies = [(self.class_names[i],100.0*(self.class_correct[i]/self.class_totals[i])) 
#                                  for i in range(len(self.class_names)) if self.class_totals[i] > 0]
#         return accuracy,class_accuracies    

class Network(nn.Module):
    def __init__(self,device=None):
        super().__init__()
        self.parallel = False
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(self.device)

    def forward(self,x):
        return self.model(x)
    def compute_loss(self,outputs,labels):
        # print(outputs.shape,labels.shape)
        # loss = self.criterion(outputs,labels)
        # print(loss)
        # return [loss]
        return [self.criterion(outputs,labels)]

    def batch_to_loss(self,data_batch):
        inputs,labels = data_batch[0],data_batch[1]
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs,labels)[0]
        return loss

    def fit(self, trainloader, validloader, cycle_len=2, num_cycles=1, max_lr=1., print_every=10,
            validate_every=1, save_best_every=1, clip=False, load_best=False,
            eval_thresh=0.5, saving_crit='loss', one_cycle_upward_epochs=None):
        # os.makedirs('saved_weights', exist_ok=True)
        weights_folder = Path('saved_weights')
        epochs = cycle_len
        optim_path = Path(self.best_model_file)
        optim_path = optim_path.stem + '_optim' + optim_path.suffix
        lr = self.optimizer.param_groups[0]['lr']
        if one_cycle_upward_epochs is not None:
            max_lr = min(max_lr,lr*10)
            start_lr = lr/100
            last_lr = lr/1000
            upward_epochs = one_cycle_upward_epochs
            remaining_epochs = epochs-upward_epochs
            max_epochs = int(np.ceil(remaining_epochs/4))
            downward_epochs = int(np.floor(remaining_epochs/3))
            last_epochs = int(np.ceil(remaining_epochs/2))
            upward_step = (max_lr-start_lr)/upward_epochs
            downward_step = -((max_lr-last_lr)/downward_epochs)
            upward_lrs = list(np.arange(start_lr,max_lr,step=upward_step))
            downward_lrs = list(np.arange(max_lr,last_lr,step=downward_step))
            lrs = upward_lrs + [max_lr]*max_epochs + downward_lrs + [last_lr]*last_epochs
            if len(lrs) < epochs:
                lrs += [last_lr]*(epochs-len(lrs))
        with mlflow.start_run() as run:
            for cycle in range(num_cycles):
                for epoch in range(epochs):
                    print(f'Cycle: {cycle+1}/{num_cycles}')
                    print('Epoch:{:3d}/{}\n'.format(epoch+1,epochs))
                    if one_cycle_upward_epochs is not None:
                        print(f'Learning Rate: {lrs[epoch]}')
                        self.optimizer.param_groups[0]['lr'] = lrs[epoch]
                    self.model = self.model.to(self.device)
                    mlflow.log_param('epochs',epochs)
                    mlflow.log_param('lr',self.optimizer.param_groups[0]['lr'])
                    mlflow.log_param('bs',trainloader.batch_size)
                    epoch_train_loss =  self.train_((epoch,epochs), trainloader, optimizer=self.optimizer,
                                                     print_every=print_every ,clip=clip)  
                            
                    if  validate_every and (epoch % validate_every == 0):
                        t2 = time.time()
                        eval_dict = self.evaluate(validloader,thresh=eval_thresh)
                        epoch_validation_loss = eval_dict['final_loss']
                        if self.parallel:
                            try:
                                epoch_train_loss = epoch_train_loss.item()
                                epoch_validation_loss = epoch_validation_loss.item()
                            except:
                                pass  
                        mlflow.log_metric('Train Loss',epoch_train_loss)
                        mlflow.log_metric('Validation Loss',epoch_validation_loss)
                        
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
                                f"Epoch validation loss: {epoch_validation_loss:.6f}"
                            )
                        if self.model_type == 'classifier':# or self.num_classes is not None:
                            epoch_accuracy = eval_dict['accuracy']
                            mlflow.log_metric('Validation Accuracy',epoch_accuracy)
                            print("Validation accuracy: {:.3f}".format(epoch_accuracy))
                            print()
                            if self.num_classes <= 5:
                                class_acc = eval_dict['class_accuracies']
                                for cl,ac in class_acc:
                                    print(f'{cl} accuracy: {ac:.4f}')
                            # print('\\'*36+'/'*36+'\n')
                            print('\\'*36+'\n')
                            if self.best_accuracy == 0. or (epoch_accuracy >= self.best_accuracy):
                                print('\n**********Updating best accuracy**********\n')
                                print('Previous best: {:.3f}'.format(self.best_accuracy))
                                print('New best: {:.3f}\n'.format(epoch_accuracy))
                                print('******************************************\n')
                                self.best_accuracy = epoch_accuracy
                                mlflow.log_metric('Best Accuracy',self.best_accuracy)
                                
                                best_model_path, optim_path = self.save_model(epoch_accuracy, epoch+1, weights_folder,
                                mlflow_saved_folder='mlflow_saved_training_models', mlflow_logged_folder='mlflow_logged_models')

                                # curr_time = str(datetime.now())
                                # curr_time = '_'+curr_time.split()[1].split('.')[0]
                                # suff = Path(self.best_model_file).suffix
                                # best_model_file = Path(self.best_model_file).stem+f'_{str(round(epoch_accuracy,2))}_{str(epoch+1)+curr_time}'
                                # best_model_path = weights_folder/(best_model_file + suff)
                                # optim_path = weights_folder/(best_model_file + '_optim' + suff)
                                # torch.save(self.model.state_dict(), best_model_path)
                                # torch.save(self.optimizer.state_dict(),optim_path)     
                                # mlflow.pytorch.log_model(self,'mlflow_logged_models')
                                # mlflow_save_path = Path('mlflow_saved_training_models')/best_model_file
                                # mlflow.pytorch.save_model(self,mlflow_save_path)
                        else:
                            if self.model_type == 'multi_label_classifier':
                                epoch_accuracy = eval_dict['accuracy']
                                mlflow.log_metric('Validation Accuracy',epoch_accuracy)
                                print("Validation accuracy: {:.3f}".format(epoch_accuracy))
                                print()
                                if self.num_classes <= 5:
                                    class_acc = eval_dict['class_accuracies']
                                    for cl,ac in class_acc:
                                        print(f'{cl} accuracy: {ac:.4f}')
                            elif self.model_type == 'super_res' or self.model_type == 'enhancement':
                                epoch_psnr = eval_dict['psnr']
                                mlflow.log_metric('Validation PSNR',epoch_psnr)
                                print("Validation psnr: {:.3f}".format(epoch_psnr))
                            # print('\\'*36+'/'*36+'\n')
                            print('\\'*36+'\n')
                            if saving_crit == 'loss':
                                if self.best_validation_loss == None or (epoch_validation_loss <= self.best_validation_loss):
                                    print('\n**********Updating best validation loss**********\n')
                                    if self.best_validation_loss is not None:
                                        print('Previous best: {:.7f}'.format(self.best_validation_loss))
                                    print('New best loss = {:.7f}\n'.format(epoch_validation_loss))
                                    print('*'*49+'\n')
                                    self.best_validation_loss = epoch_validation_loss
                                    mlflow.log_metric('Best Loss',self.best_validation_loss)

                                    best_model_path, optim_path = self.save_model(epoch_validation_loss, epoch+1, weights_folder,
                                    mlflow_saved_folder='mlflow_saved_training_models', mlflow_logged_folder='mlflow_logged_models')

                                    # optim_path = Path(self.best_model_file)
                                    # optim_path = optim_path.stem + '_optim' + optim_path.suffix
                                    # torch.save(self.model.state_dict(),self.best_model_file)
                                    # torch.save(self.optimizer.state_dict(),optim_path)     
                                    # mlflow.pytorch.log_model(self,'mlflow_logged_models')
                                    # curr_time = str(datetime.now())
                                    # curr_time = '_'+curr_time.split()[1].split('.')[0]
                                    # mlflow_save_path = Path('mlflow_saved_training_models')/\
                                    #     (Path(self.best_model_file).stem+'_{}_{}'.format(str(round(epoch_validation_loss,3)),str(epoch+1)+curr_time))
                                    # mlflow.pytorch.save_model(self,mlflow_save_path)
                            elif saving_crit == 'psnr':
                                if self.best_psnr == None or (epoch_psnr >= self.best_psnr):
                                    print('\n**********Updating best psnr**********\n')
                                    if self.psnr is not None:
                                        print('Previous best: {:.7f}'.format(self.best_psnr))
                                    print('New best psnr = {:.7f}\n'.format(epoch_psnr))
                                    print('*'*49+'\n')
                                    self.best_psnr = epoch_psnr
                                    mlflow.log_metric('Best Psnr',self.best_psnr)

                                    best_model_path, optim_path = self.save_model(epoch_psnr, epoch+1, weights_folder,
                                    mlflow_saved_folder='mlflow_saved_training_models', mlflow_logged_folder='mlflow_logged_models')

                                    # optim_path = Path(self.best_model_file)
                                    # optim_path = optim_path.stem + '_optim' + optim_path.suffix
                                    # torch.save(self.model.state_dict(),self.best_model_file)
                                    # torch.save(self.optimizer.state_dict(),optim_path)     
                                    # mlflow.pytorch.log_model(self,'mlflow_logged_models')
                                    # curr_time = str(datetime.now())
                                    # curr_time = '_'+curr_time.split()[1].split('.')[0]
                                    # mlflow_save_path = Path('mlflow_saved_training_models')/\
                                    #     (Path(self.best_model_file).stem+'_{}_{}'.format(str(round(epoch_psnr,3)),str(epoch+1)+curr_time))
                                    # mlflow.pytorch.save_model(self,mlflow_save_path)
                            elif saving_crit == 'accuracy':
                                if self.best_accuracy == 0. or (epoch_accuracy >= self.best_accuracy):
                                    print('\n**********Updating best accuracy**********\n')
                                    print('Previous best: {:.3f}'.format(self.best_accuracy))
                                    print('New best: {:.3f}\n'.format(epoch_accuracy))
                                    print('******************************************\n')
                                    self.best_accuracy = epoch_accuracy
                                    mlflow.log_metric('Best Accuracy',self.best_accuracy)

                                    best_model_path, optim_path = self.save_model(epoch_accuracy, epoch+1, weights_folder,
                                    mlflow_saved_folder='mlflow_saved_training_models', mlflow_logged_folder='mlflow_logged_models')

                                    # optim_path = Path(self.best_model_file)
                                    # optim_path = optim_path.stem + '_optim' + optim_path.suffix
                                    # torch.save(self.model.state_dict(),self.best_model_file)
                                    # torch.save(self.optimizer.state_dict(),optim_path)     
                                    # mlflow.pytorch.log_model(self,'mlflow_logged_models')
                                    # curr_time = str(datetime.now())
                                    # curr_time = '_'+curr_time.split()[1].split('.')[0]
                                    # mlflow_save_path = Path('mlflow_saved_training_models')/\
                                    #     (Path(self.best_model_file).stem+'_{}_{}'.format(str(round(epoch_accuracy,2)),str(epoch+1)+curr_time))
                                    # mlflow.pytorch.save_model(self,mlflow_save_path)

                        self.train()
        torch.cuda.empty_cache()
        if load_best:
            try:
                print('\nLoaded best model\n')
                self.model.load_state_dict(torch.load(best_model_path))
                self.optimizer.load_state_dict(torch.load(optim_path))
                # os.remove(self.best_model_file)
                # os.remove(optim_path)
            except:
                pass    

    def train_(self, e, trainloader, optimizer=None, print_every=10, clip=False):

        self.train()
        if optimizer is None:
            optimizer = self.optimizer
        epoch,epochs = e
        t0 = time.time()
        t1 = time.time()
        batches = 0
        running_loss = 0.
        for data_batch in trainloader:
            batches += 1
            # inputs,labels = data_batch[0],data_batch[1]
            # inputs = inputs.to(self.device)
            # labels = labels.to(self.device)
            # outputs = self.forward(inputs)
            # loss = self.compute_loss(outputs,labels)[0]
            loss = self.batch_to_loss(data_batch)
            optimizer.zero_grad()
            if self.parallel:
                loss.sum().backward()
                loss = loss.sum()
            else:    
                loss.backward()
                loss = loss.item()
            if clip:
                nn.utils.clip_grad_norm_(self.model.parameters(),1.)
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

    def evaluate(self,dataloader,metric='accuracy',thresh=0.5):
        
        running_loss = 0.
        classifier = None

        if self.model_type == 'classifier' or self.model_type == 'multi_label_classifier':
            classifier = Classifier(self.class_names)
        # elif self.model_type == 'multi_label_classifier':
        #     classifier = MultiLabelClassifier(self.class_names)

        y_pred = []
        y_true = []

        self.eval()
        rmse_ = 0.
        with torch.no_grad():
            for data_batch in dataloader:
                inputs,labels = data_batch[0],data_batch[1]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(inputs)
                loss = self.compute_loss(outputs,labels)[0]
                # loss = self.batch_to_loss(data_batch)
                if self.parallel:
                    running_loss += loss.sum()
                    outputs = parallel.gather(outputs,self.device)
                else:        
                    running_loss += loss.item()
                if classifier is not None:
                    if self.model_type == 'classifier':
                        classifier.update_accuracies(outputs,labels)
                        try:
                            y_true.extend(list(labels.squeeze(0).cpu().numpy()))
                            _, preds = torch.max(torch.exp(outputs), 1)
                            y_pred.extend(list(preds.cpu().numpy()))
                        except:
                            pass
                    else:
                        classifier.update_multi_accuracies(outputs,labels,thresh)
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
            try:
                ret['report'] = classification_report(y_true,y_pred,target_names=self.class_names)
                ret['confusion_matrix'] = confusion_matrix(y_true,y_pred)
                try:
                    ret['roc_auc_score'] = roc_auc_score(y_true,y_pred)
                except:
                    pass
            except:
                pass
        return ret
   
    def classify(self,inputs,thresh = 0.4,class_names = None):#,show = False,mean = None,std = None):
        if class_names is None:
            class_names = self.class_names
        outputs = self.predict(inputs)
        if self.model_type == 'classifier':
            try:    
                _, preds = torch.max(torch.exp(outputs), 1)
                # _, preds = torch.max(outputs, 1)
            except:
                _, preds = torch.max(torch.exp(outputs.unsqueeze(0)), 1)
        else:
            outputs = outputs.sigmoid()
            preds = (outputs > thresh).nonzero().squeeze(1)
        class_preds = [str(class_names[p]) for p in preds]
        # imgs = batch_to_imgs(inputs.cpu(),mean,std)
        # if show:
            # plot_in_row(imgs,titles=class_preds)
        return class_preds

    def predict(self,inputs,actv = None):
        self.eval()
        self.model.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.forward(inputs)
        if actv is not None:
            return actv(outputs)
        return outputs
    
    def pred_tta(self,data_dir,data,tta,actv=None,thresh=0.5,channels=3):
        preds = []
        labels = []
        for i in range(len(data)):
            img_path = os.path.join(data_dir,data.iloc[i, 0])
            if self.channels == 3:
                img = bgr2rgb(cv2.imread(str(img_path)))
            else:    
                img = cv2.imread(str(img_path),0)
            tta_preds = []
            for t in tta:
                tfms = albu.Compose(t)
                img_t = tfms(image=img)['image'].unsqueeze(0).unsqueeze(0)
                p = (self.predict(img_t,actv) > thresh).squeeze(0)
                tta_preds.append(p)
            pred = torch.stack(tta_preds).mode(dim=0)[0]
            label = data.iloc[i,1]
            labels.append(label)
            preds.append(pred)
        preds = torch.stack(preds).float()
        labels = torch.stack(labels).float()
        return preds,labels
    
    def evaluate_tta(self,data_dir,data,tta,actv=None,thresh=0.5,channels=3,metric='accuracy'):

        classifier = None
        if self.model_type == 'classifier' or self.model_type == 'multi_label_classifier':
            classifier = Classifier(self.class_names)
        # elif self.model_type == 'multi_label_classifier':
            # classifier = MultiLabelClassifier(self.class_names)

        y_pred = []
        y_true = []

        self.eval()
        with torch.no_grad():
            outputs,labels = self.pred_tta(data_dir,data,tta,actv,thresh,channels)
            labels = labels.to(self.device)
            outputs = outputs.to(self.device)
            loss = self.compute_loss(outputs,labels)[0]
            running_loss = loss.item()
            if classifier is not None:
                if self.model_type == 'classifier':
                    classifier.update_tta_accuracies(outputs,labels)
                    try:
                        y_true.extend(list(labels.squeeze(0).cpu().numpy()))
                        _, preds = torch.max(torch.exp(outputs), 1)
                        y_pred.extend(list(preds.cpu().numpy()))
                    except:
                        pass
                else:
                    classifier.update_multi_tta_accuracies(outputs,labels)
            elif metric == 'rmse':
                rmse_ = rmse(outputs,labels).cpu().numpy()
        self.train()
        ret = {}
        if metric == 'rmse':
            print('Total rmse: {:.3f}'.format(rmse_))
            ret['final_rmse'] = rmse_/len(labels)
        ret['final_loss'] = running_loss/len(labels)
        
        if classifier is not None:
            ret['accuracy'],ret['class_accuracies'] = classifier.get_final_accuracies()
            try:
                ret['report'] = classification_report(y_true,y_pred,target_names=self.class_names)
                ret['confusion_matrix'] = confusion_matrix(y_true,y_pred)
                try:
                    ret['roc_auc_score'] = roc_auc_score(y_true,y_pred)
                except:
                    pass
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
            if batch_num % 100 == 0:
                print(batch_num)
            # inputs,labels = data_batch[0],data_batch[1]
            # inputs = inputs.to(self.device)
            # labels = labels.to(self.device)
            # outputs = self.forward(inputs)
            # loss = self.compute_loss(outputs,labels)[0]
            loss = self.batch_to_loss(data_batch)
            optimizer.zero_grad()
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
                
    def set_criterion(self, criterion):
        if criterion:
            self.criterion = criterion
        
    def set_optimizer(self,params,optimizer_name='adam',lr=0.003):
        if optimizer_name:
            optimizer_name = optimizer_name.lower()
            if optimizer_name == 'adam':
                print('Setting optimizer: Adam')
                self.optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
                self.optimizer_name = optimizer_name
            elif optimizer_name == 'sgd':
                print('Setting optimizer: SGD')
                self.optimizer = optim.SGD(params,lr=lr)
                self.optimizer_name = optimizer_name
            elif optimizer_name == 'adadelta':
                print('Setting optimizer: AdaDelta')
                # self.optimizer = optim.Adadelta(params)
                self.optimizer = optim.Adadelta(params,lr=lr)
                self.optimizer_name = optimizer_name
            
    def set_model_params(self,
                         params = None,
                         criterion = nn.CrossEntropyLoss(),
                         optimizer_name = 'sgd',
                         lr = 0.01,
                         dropout_p = 0.45,
                         model_name = 'resnet50',
                         model_type = 'classifier',
                         best_accuracy = 0.,
                         best_validation_loss = None,
                         best_model_file = 'best_model_file.pth'):        
        if params is None:
            params = self.model.parameters()
        self.set_criterion(criterion)
        self.optimizer_name = optimizer_name
        self.set_optimizer(params,optimizer_name,lr=lr)
        self.lr = lr
        self.dropout_p = dropout_p
        self.model_name =  model_name
        self.model_type = model_type
        self.best_accuracy = best_accuracy
        self.best_validation_loss = best_validation_loss
        self.best_model_file = best_model_file
    
    def get_model_params(self):
        params = {}
        params['device'] = self.device
        params['model_type'] = self.model_type
        params['model_name'] = self.model_name
        params['optimizer_name'] = self.optimizer_name
        params['criterion'] = self.criterion
        params['lr'] = self.lr
        params['dropout_p'] = self.dropout_p
        params['best_accuracy'] = self.best_accuracy
        params['best_validation_loss'] = self.best_validation_loss
        params['best_model_file'] = self.best_model_file
        return params
    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_model(self, crit='', epoch='', weights_folder='weights_folder',
                   mlflow_saved_folder='mlflow_saved_training_models', mlflow_logged_folder='mlflow_logged_models'):
        weights_folder = Path(weights_folder)
        os.makedirs(weights_folder, exist_ok=True)
        if type(epoch) != str:
            epoch = str(epoch)
        if type(crit) != str:
            crit = str(round(crit,3))
        curr_time = str(datetime.now())
        curr_time = '_'+curr_time.split()[1].split('.')[0]
        suff = Path(self.best_model_file).suffix
        best_model_file = Path(self.best_model_file).stem+f'_{crit}_{epoch+curr_time}'
        best_model_path = weights_folder/(best_model_file + suff)
        optim_path = weights_folder/(best_model_file + '_optim' + suff)
        torch.save(self.model.state_dict(), best_model_path)
        torch.save(self.optimizer.state_dict(),optim_path)     
        mlflow.pytorch.log_model(self,mlflow_logged_folder)
        mlflow_save_path = Path(mlflow_saved_folder)/best_model_file
        mlflow.pytorch.save_model(self,mlflow_save_path)
        return best_model_path, optim_path

    def parallelize(self):
        self.parallel = True
        self.model = DataParallelModel(self.model)
        self.criterion  = DataParallelCriterion(self.criterion)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
