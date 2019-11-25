from dai_imports import*
from data_processing import get_img_stats

def display_img_actual_size(im_data,title = ''):
    dpi = 80
    height, width, depth = im_data.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, cmap='gray')
    plt.title(title,fontdict={'fontsize':25})
    plt.show()

def plt_show(im,cmap=None):
    plt.imshow(im,cmap=cmap)
    plt.show()

def plt_load(path,show = False, cmap = None):
    img = plt.imread(path)
    if show:
        plt_show(img,cmap=cmap)
    return img    

def denorm_img_general(inp,mean=None,std=None):
    inp = inp.numpy()
    inp = inp.transpose((1, 2, 0))
    if mean is None:
        mean = np.mean(inp)
    if std is None:    
        std = np.std(inp)
    inp = std * inp + mean
    inp = np.clip(inp, 0., 1.)
    return inp 

def bgr2rgb(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def rgb2bgr(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

def gray2rgb(img):
    return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

def rgb2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

def bgra2rgb(img):
    return cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)

def plot_in_row(imgs,figsize = (20,20),rows = None,columns = None,titles = [],fig_path = 'fig.png',cmap = None):
    fig=plt.figure(figsize=figsize)
    if len(titles) == 0:
        titles = ['image_{}'.format(i) for i in range(len(imgs))]
    if not rows:
        rows = 1
        if columns:
            rows = len(imgs)//columns    
    if not columns:    
        columns = len(imgs)
        if rows:
            columns = len(imgs)//rows
    for i in range(1, columns*rows +1):
        img = imgs[i-1]
        fig.add_subplot(rows, columns, i, title = titles[i-1])
        plt.imshow(img,cmap=cmap)
    fig.savefig(fig_path)    
    plt.show()
    return fig

def denorm_tensor(x,img_mean,img_std):
    if x.dim() == 3:
        x.unsqueeze_(0)
    x[:, 0, :, :] = x[:, 0, :, :] * img_std[0] + img_mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * img_std[1] + img_mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * img_std[2] + img_mean[2]
    return x

def tensor_to_img(t):
    if t.dim() > 3:
        return [np.transpose(t_,(1,2,0)) for t_ in t]
    return np.transpose(t,(1,2,0))

def smooth_labels(labels,eps=0.1):
    if len(labels.shape) > 1:
        length = len(labels[0])
    else:
        length = len(labels)
    labels = labels * (1 - eps) + (1-labels) * eps / (length - 1)
    return labels

# def get_img_stats(dataset,tfms,channels,sz=1.):

#     print('Calculating mean and std of the data for standardization. Might take some time, depending on the data size.')

#     size = int(len(dataset)*sz)
#     imgs = []
#     for i,img in enumerate(dataset):
#         if i > size:
#             break
#         img = tfms(image=img)['image']
#         if channels == 1:
#             img.unsqueeze_(0)
#         imgs.append(img)
#     imgs_ = torch.stack(imgs,dim=3)
#     imgs_ = imgs_.view(channels,-1)
#     imgs_mean = imgs_.mean(dim=1)
#     imgs_std = imgs_.std(dim=1)
#     del imgs
#     del imgs_
#     print('Done')
#     return imgs_mean,imgs_std

class imgs_to_batch_dataset(Dataset):
    
    def __init__(self, data, transforms_ = None, channels = 3):
        super(imgs_to_batch_dataset, self).__init__()
        self.data = data
        self.transforms_ = transforms_
        self.tfms = None
        self.channels = channels
        assert transforms_ is not None, print('Please pass some transforms.')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        try:
            img_path = self.data.iloc[index, 0]
            if self.channels == 3:
                img = bgr2rgb(cv2.imread(str(img_path)))
            else:    
                img = cv2.imread(str(img_path),0)
        except:
            img = self.data.iloc[index, 0]
        self.tfms = albu.Compose(self.transforms_)
        x = self.tfms(image=img)['image']
        if self.channels == 1:
            x = x.unsqueeze(0)
        return x,img_path

def imgs_to_batch(paths = [], imgs = [], bs = 1, size = None, norm = False, img_mean = None, img_std = None,
                  stats_percentage = 1., channels = 3, num_workers = 6):
    if len(paths) > 0:
        data = pd.Dataframe({'Images':paths})
    elif len(imgs) > 0:
        data = pd.Dataframe({'Images':imgs})
    tfms = [AT.ToTensor()]
    if size:
        tfms.insert(0,albu.Resize(size[0],size[1],interpolation=0))        
    if norm:
        if img_mean is None:
            norm_tfms = albu.Compose(tfms)
            frac_data = data.sample(frac = stats_percentage).reset_index(drop=True).copy()
            temp_dataset = imgs_to_batch_dataset(data = frac_data, transforms_ = norm_tfms, channels = channels)
            img_mean,img_std = get_img_stats(temp_dataset,channels)
        tfms.insert(1,albu.Normalize(img_mean,img_std))
    tfms = albu.Compose(tfms)
    image_dataset = imgs_to_batch_dataset(data = data, transforms_ = tfms, channels = channels)
    loader = DataLoader(image_dataset, batch_size = bs, shuffle=True, num_workers = num_workers)
    return image_dataset,loader

# def imgs_to_batch_old(paths = [],imgs = [], size = None, smaller_factor = None, enlarge_factor = None, mean = None, std = None,
#                   stats_percentage = 1.,show = False, norm = False, bgr_to_rgb = False, device = None, channels = 3):
#     tfms = [AT.ToTensor()]    
#     if len(paths) > 0:
#         if channels == 3:
#             bgr_to_rgb = True
#             imgs = []
#             for p in paths:
#                 imgs.append(cv2.imread(str(p)))
#         elif channels == 1:
#             imgs = []
#             for p in paths:
#                 imgs.append(cv2.imread(str(p),0))
#     if size:
#         tfms.insert(0,albu.Resize(size[0],size[1],interpolation=0))        
#     if norm:
#         norm_tfms = albu.Compose(tfms)
#         if mean is None:
#             mean,std = get_img_stats(imgs,norm_tfms,channels,stats_percentage)
#         tfms.insert(1,albu.Normalize(mean,std))
#     for i,img in enumerate(imgs):
#         if bgr_to_rgb:
#             img = bgr2rgb(img)
#         if show:
#             cmap = None
#             if channels == 1:
#                 cmap = 'gray'
#             try:
#                 plt_show(img,cmap=cmap)   
#             except:
#                 plt_show(img,cmap=cmap)
        
#         transform = albu.Compose(tfms)
#         x = transform(image=img)['image']
#         if channels == 1:
#             x.unsqueeze_(0)
#         imgs[i] = x
#     batch = torch.stack(imgs, dim=0)
#     if device is not None:
#         batch = batch.to(device)
#     return batch

def to_batch(paths = [],imgs = [], size = None, channels=3):
    if len(paths) > 0:
        imgs = []
        for p in paths:
            if channels==3:
                img = bgr2rgb(cv2.imread(p))
            elif channels==1:
                img = cv2.imread(p,0)
            imgs.append(img)
    for i,img in enumerate(imgs):
        if size:
            img = cv2.resize(img, size)
        img =  img.transpose((2,0,1))
        imgs[i] = img
    return torch.from_numpy(np.asarray(imgs)).float()    

def batch_to_imgs(batch,mean=None,std=None):
    imgs = []
    for i in batch:
        imgs.append(denorm_img_general(i,mean,std))
    return imgs    

def mini_batch(dataset,bs,start=0):
    imgs = torch.Tensor(bs,*dataset[0][0].shape)
    s = dataset[0][1].shape
    if len(s) > 0:
        labels = torch.Tensor(bs,*s)
    else:    
        labels = torch.Tensor(bs).int()
    for i in range(start,bs+start):
        b = dataset[i]
        imgs[i-start] = b[0]
        labels[i-start] = tensor(b[1])
    return imgs,labels    

def get_optim(optimizer_name,params,lr):
    if optimizer_name.lower() == 'adam':
        return optim.Adam(params=params,lr=lr)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(params=params,lr=lr)
    elif optimizer_name.lower() == 'adadelta':
        return optim.Adadelta(params=params)

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

# class conv_block(in_channels,out_channels,kernel,stride,padding,relu):
#     m = [nn.Conv2d(in_channels,out_channels,kernel_size=5,padding=2),nn.PReLU()]
#     if in_channels == out_channels:
#         m.append(nn.MaxPool2d(2))
#     return nn.Sequential(*m)

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, relu=True, bn=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
    if relu: layers.append(nn.ReLU(True))
    if bn: layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

class WeightedMultilabel(nn.Module):
    def __init__(self, weights):
        super(WeightedMultilabel,self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.weights = weights.unsqueeze(0)

    def forward(self,outputs, targets):
        loss = torch.sum(self.loss(outputs, targets) * self.weights) 
        return loss

class MultiConcatHeader(nn.Module):
    def __init__(self,fc1,fc2):
        super(MultiConcatHeader, self).__init__()
        self.fc1 = fc1
        self.fc2 = fc2
    def forward(self,x):
        single_label = self.fc1(x)
        single_index = torch.argmax(torch.softmax(single_label,1),dim=1).float().unsqueeze(1)
        # print(flatten_tensor(x).shape,single_index.shape)
        multi_input = torch.cat((flatten_tensor(x),single_index),dim=1)
        multi_label = self.fc2(multi_input)
        return single_label,multi_label

class MultiSeparateHeader(nn.Module):
    def __init__(self,fc1,fc2):
        super(MultiSeparateHeader, self).__init__()
        self.fc1 = fc1
        self.fc2 = fc2
    def forward(self,x):
        single_label = self.fc1(x)
        multi_label = self.fc2(x)
        return single_label,multi_label

class Printer(nn.Module):
    def forward(self,x):
        print(x.size())
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove() 

DAI_AvgPool = nn.AdaptiveAvgPool2d(1)

def flatten_tensor(x):
    return x.view(x.shape[0],-1)

def rmse(inputs,targets):
    return torch.sqrt(torch.mean((inputs - targets) ** 2))

def psnr(mse):
    return 10 * math.log10(1 / mse)

def get_psnr(inputs,targets):
    mse_loss = F.mse_loss(inputs,targets)
    return 10 * math.log10(1 / mse_loss)

def remove_bn(s):
    for m in s.modules():
        if isinstance(m,nn.BatchNorm2d):
            m.eval()

def dice(input, targs, iou = False, eps = 1e-8):
    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    intersect = (input * targs).sum(dim=1).float()
    union = (input+targs).sum(dim=1).float()
    if not iou: l = 2. * intersect / union
    else: l = intersect / (union-intersect+eps)
    l[union == 0.] = 1.
    return l.mean()

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = F.sigmoid(pred.argmax(dim=1).view(num, -1).float())  # Flatten
    m2 = F.sigmoid(target.view(num, -1).float())  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def save_frames(video_path, num_frames = None, output_path = '', frame_name = ''):

    if len(frame_name) == 0:
        frame_name = Path(video_path).name[:-4]
    vs = cv2.VideoCapture(video_path)
    frame_number = 0
    frames = []
    while True:  
        if num_frames:
            if frame_number >= num_frames:
                break
        # Grab a frame from the video stream
        (grabbed, frame) = vs.read()
        if grabbed:
            frame_number+=1
        # If the frame was not grabbed, then we have reached the end of the video
        if not grabbed:
            break
        frame = bgr2rgb(frame)
        frames.append(frame)
        if len(output_path) > 0:
            os.makedirs(output_path,exist_ok=True)
            plt.imsave(Path(output_path)/(frame_name+'_frame_{}.png'.format(frame_number)),frame)
    vs.release()
    return frames

def expand_rect(left,top,right,bottom,H,W, margin = 15):
    if top >= margin:
        top -= margin
    if left >= margin:
        left -= margin
    if bottom <= H-margin:
        bottom += margin
    if right <= W-margin:
        right += margin
    return left,top,right,bottom

def show_landmarks(image, landmarks):
    landmarks = np.array(landmarks)    
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

def path_list(path,sort = False):
    if sort:
        return sorted(list(Path(path).iterdir()))
    return list(Path(path).iterdir())

def sorted_paths(path,reverse = True):
    return sorted(path_list(path),key = lambda x: x.stat().st_ctime, reverse=reverse)

def process_landmarks(lm):
    lm_keys = ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip',
               'left_eye', 'right_eye', 'top_lip', 'bottom_lip']
    landmarks = []
    check = False
    if len(lm) > 0:
        lm = lm[0]
        if sum(x < 0 for x in list(sum([list(sum(x,())) for x in list(lm.values())],[]))) == 0:
            check = True
            marks = []
            for k in lm_keys:
                marks += lm[k]
            landmarks+=(list(sum(marks,())))
    return landmarks,check

def is_confused_or_wrong(p,l,wrong_th=0.65):
    for i in range(len(p)):
        if ((abs(l[i].item() - p[i].item()) > wrong_th) or (p[i].item() >= 0.4 and p[i].item() <= 0.6)):
            return True
    return False

def get_confused_or_wrong_names(preds,labels,names,wrong_th=0.65):
    retrain = []
    for i,p in enumerate(preds):
        label,name = labels[i],names[i]
        if is_confused_or_wrong(p,label,wrong_th):
            retrain.append(name)
    return retrain

def is_confused(p):
    for x in p:
        if (x.item() >= 0.4 and x.item() <= 0.6):
            return True
    return False

def is_wrong(p,l,wrong_th=0.65):
    for x,y in zip(l,p):
        if (abs(x.item() - y.item()) > wrong_th):
            return True
    return False

def get_wrong(preds,labels,names,wrong_th=0.65):
    retrain = []
    for i,p in enumerate(preds):
        label,name = labels[i],names[i]
        if is_wrong(p,label,wrong_th):
            retrain.append(name)
    return retrain

def get_confused(preds,labels,names):
    retrain = []
    for p,name in zip(preds,names):
        if is_confused(p):
            retrain.append(name)
    return retrain

def get_random_retrain(retrain,names):
    for n in names:
        if random.random() > 0.1 and n not in retrain:
            retrain.append(n)
    return retrain

def get_retrain(model,loader,wrong_th=0.65):
    retrain = []
    for batch in loader:
        preds,labels,names = torch.sigmoid(model.predict(batch[0])),batch[1],batch[2]
        retrain += get_random_retrain(get_confused_or_wrong_names(preds,labels,names,wrong_th),names)
    return retrain

def get_wrong_class(preds,labels,names,class_idx,wrong_th=0.75):
    retrain = []
    for i,pred in enumerate(preds):
        label,name = labels[i],names[i]
        if ((label[class_idx].item() - pred[class_idx].item()) > wrong_th):
            retrain.append(name)
    return retrain

def get_retrain_wrong_class(model,loader,class_idx,wrong_th=0.75):
    retrain = []
    for batch in loader:
        preds,labels,names = torch.sigmoid(model.predict(batch[0])),batch[1],batch[2]
        retrain += get_wrong_class(preds,labels,names,class_idx,wrong_th)
    return retrain

def get_confused_class(preds,names,class_idx):
    retrain = []
    for name,pred in zip(names,preds):
        if pred[class_idx].item() >= 0.4 and pred[class_idx].item() <= 0.6:
            retrain.append(name)
    return retrain

def get_retrain_confused_class(model,loader,class_idx):
    retrain = []
    for batch in loader:
        preds,names = torch.sigmoid(model.predict(batch[0])),batch[2]
        retrain += get_confused_class(preds,names,class_idx)
    return retrain

def get_all_zeros(preds,names):
    retrain = []
    for name,pred in zip(names,preds):
        if sum(pred).item() == 0:
            retrain.append(name)
    return retrain

def get_retrain_all_zeros(model,loader):
    retrain = []
    for batch in loader:
        preds,names = torch.sigmoid(model.predict(batch[0])),batch[2]
        retrain += get_all_zeros(preds,names)
    return retrain

def get_class(labels,names,class_idx):
    retrain = []
    for label,name in zip(labels,names):
        if label[class_idx] == 1:
            retrain.append(name)
    return retrain

def get_retrain_class(loader,class_idx):
    retrain = []
    for batch in loader:
        labels,names = batch[1],batch[2]
        retrain += get_class(labels,names,class_idx)
    return retrain
