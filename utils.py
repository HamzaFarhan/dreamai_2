from dreamai import pyflow
from dreamai.dai_imports import*
from dreamai.util_classes import*
from dreamai.data_processing import get_img_stats

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

def img_on_bg(img, bg, x_factor=1/2, y_factor=1/2):

    img_h, img_w = img.shape[:2]
    background = Image.fromarray(bg)
    bg_w, bg_h = background.size
    offset = (int((bg_w - img_w) * x_factor), int((bg_h - img_h) * y_factor))
    background.paste(Image.fromarray(img), offset)
    img = np.array(background)
    return img

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

def img_float_to_int(img):
    return np.clip((np.array(img)*255).astype(np.uint8),0,255)

def img_int_to_float(img):
    return np.clip((np.array(img)/255).astype(np.float),0.,1.)

def adjust_lightness(color, amount=1.2):
    color = img_int_to_float(color)
    c = colorsys.rgb_to_hls(*color)
    c = np.array(colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2]))
    return img_float_to_int(c)

def albu_resize(img, h, w, interpolation=1):
    rz = albu.Resize(h, w, interpolation)
    return rz(image=img)['image']

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
    return np.array(np.transpose(t,(1,2,0)))

def smooth_labels(labels,eps=0.1):
    if len(labels.shape) > 1:
        length = len(labels[0])
    else:
        length = len(labels)
    labels = labels * (1 - eps) + (1-labels) * eps / (length - 1)
    return labels

def get_flow(im1, im2):

    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    im1 = im1.copy(order='C')
    im2 = im2.copy(order='C')
    # print(im1.shape, im2.shape)
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    #flow = rescale_flow(flow,0,1)
    return flow

def to_tensor(x):
    t = AT.ToTensorV2()
    if type(x) == list:
        return [t(image=i)['image'] for i in x]
    return t(image=x)['image']

def imgs_to_batch(paths = [], imgs = [], bs = 1, size = None, norm = False, img_mean = None, img_std = None,
                  stats_percentage = 1., channels = 3, num_workers = 6):
    if len(paths) > 0:
        data = pd.DataFrame({'Images':paths})
    elif len(imgs) > 0:
        data = pd.DataFrame({'Images':imgs})
    tfms = [AT.ToTensor()]
    if norm:
        if img_mean is None:
            norm_tfms = albu.Compose(tfms)
            frac_data = data.sample(frac = stats_percentage).reset_index(drop=True).copy()
            temp_dataset = imgs_to_batch_dataset(data = frac_data, transforms_ = norm_tfms, channels = channels)
            img_mean,img_std = get_img_stats(temp_dataset,channels)
        tfms.insert(0,albu.Normalize(img_mean,img_std))
    if size:
        tfms.insert(0,albu.Resize(size[0],size[1],interpolation=0))        
    tfms = albu.Compose(tfms)
    image_dataset = imgs_to_batch_dataset(data = data, transforms_ = tfms, channels = channels)
    if size is None:
        loader = None
    else:
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
        if mean is not None:
            imgs.append(denorm_img_general(i,mean,std))
        else:
            imgs.append(i)
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

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, relu=True, bn=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
    if relu: layers.append(nn.ReLU(True))
    if bn: layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers) 

def cnn_input(o,k,s,p):
    return ((o*s)-s)+k-(2*p) 

def cnn_output(w,k,s,p):
    return np.floor(((w-k+(2*p))/s))+1

def cnn_stride(w,o,k,p):
    return np.floor((w-k+(2*p))/(o-1))

def cnn_padding(w,o,k,s):
    return np.floor((((o*s)-s)-w+k)/2 )

DAI_AvgPool = nn.AdaptiveAvgPool2d(1)

def flatten_tensor(x):
    return x.view(x.shape[0],-1)

def flatten_list(l):
    return sum(l, [])

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

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def get_frames(video_path, start = None, stop = None):

    if start is None:
        start = 0
    vs = cv2.VideoCapture(str(video_path))
    vs.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_number = start
    frames = []
    while True:  
        if stop is not None:
            if frame_number >= stop:
                break
        (grabbed, frame) = vs.read()
        if grabbed:
            frame_number+=1
        if not grabbed:
            break
        frame = bgr2rgb(frame)
        frames.append(frame.astype(float)/255.)
    frames = frames[start:]
    vs.release()
    return frames

def save_imgs(imgs, dest_path = '', img_name = ''):

    if len(img_name) == 0:
        img_name = 'img'
    dest_path = Path(dest_path)
    os.makedirs(dest_path, exist_ok=True)
    for i,img in enumerate(imgs):
        plt.imsave(str(dest_path/f'{img_name}_{i}.png'), img)

def frames_to_vid(frames=[], frames_folder='', output_path='', fps=30):

    os.makedirs(Path(output_path).absolute().parent, exist_ok=True)
    if len(frames) == 0:
        frames_path = path_list(Path(frames_folder))
        first_frame = bgr2rgb(cv2.imread(str(frames_path[0])))
        height, width, _ = first_frame[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        for frame_path in frames_path:
            frame = bgr2rgb(cv2.imread(str(frame_path)))
            out.write(bgr2rgb(np.uint8(frame*255)))
        out.release()
    else:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for frame in frames:
            out.write(bgr2rgb(np.uint8(frame*255)))
        out.release()

def split_video(video_path, start=0, stop=None, output_path='split_video.mp4', fps=30):

    os.makedirs(Path(output_path).absolute().parent, exist_ok=True)
    if start is None:
        start = 0
    vs = cv2.VideoCapture(str(video_path))
    vs.set(cv2.CAP_PROP_POS_FRAMES, start)
    (grabbed, frame) = vs.read()
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_number = start
    while True:  
        if stop is not None:
            if frame_number >= stop:
                break
        (grabbed, frame) = vs.read()
        if grabbed:
            frame_number+=1
        if not grabbed:
            break
        frame = bgr2rgb(frame).astype(float)/255.
        out.write(bgr2rgb(np.uint8(frame*255)))
    out.release()
    vs.release()

def add_text(img, text, x_factor=2, y_factor=2, font=cv2.FONT_HERSHEY_SIMPLEX, scale=5.,
                color='white', thickness=10):
    color = color_to_rgb(color)
    textsize = cv2.getTextSize(text, font, scale, thickness)[0]
    textX = ((img.shape[1] - textsize[0]) // x_factor)
    textY = img.shape[0] - (((img.shape[0] - textsize[1]) // y_factor))
    img = cv2.putText(img, text, (textX, textY), font, scale, color, thickness)
    return img

def add_text_pil(img, text=['DreamAI'], x=None, y=None, font='verdana', font_size=None,
                 color='white', stroke_width=0, stroke_fill='blue', align='center'):

    if type(text) == str:
        text = [text]
    x_,y_ = x,y
    img = Image.fromarray(img)
    for i,txt in enumerate(text):
        if font_size is None:
            font_size = img.size[1]
            s = img.size
            while sum(np.array(s) < img.size) < 2:
                font_size -= int(img.size[1]/10)
                fnt = ImageFont.truetype(get_font(font), font_size)
                d = ImageDraw.Draw(img)
                s = fnt.getsize(txt)
        else:
            fnt = ImageFont.truetype(get_font(font), font_size)
            d = ImageDraw.Draw(img)
            s = fnt.getsize(txt)
        # stroke_width = font_size//30
        offset = i * int(s[1]*1.5)
        if x_ is None:
            x = (img.size[0]//2) - (s[0]//2)
        if y_ is None:
            y = (img.size[1]//2) - (s[1]//2) - (s[1]*(len(text)-1)) + offset
        d.text((x, y), txt, font=fnt, fill=color, align=align, stroke_width=stroke_width, stroke_fill=stroke_fill)
    img = np.array(img)
    return img

def remove_from_list(l, r):
    for x in r:
        if x in l:
            l.remove(x)
    return l

def num_common(l1, l2):
    return len(list(set(l1).intersection(set(l2))))

def max_n(l, n=3):
    a = np.array(l)
    idx = heapq.nlargest(n, range(len(a)), a.take)
    return idx, a[idx]

def k_dominant_colors(img, k):

    img = img.reshape((img.shape[0] * img.shape[1], 3))
    clt = KMeans(n_clusters = k)
    clt.fit(img)
    return clt.cluster_centers_

def solid_color_img(shape=(300,300,3), color='black'):
    image = np.zeros(shape, np.uint8)
    color = color_to_rgb(color)
    image[:] = color
    return image

def color_to_rgb(color):
    if type(color) == str:
        return np.array(colors.to_rgb(color)).astype(int)*255
    return color

def get_font(font):
    fonts = [f.fname for f in matplotlib.font_manager.fontManager.ttflist if ((font.lower() in f.name.lower()) and not ('italic' in f.name.lower()))]
    if len(fonts) == 0:
        print(f'"{font.capitalize()}" font not found.')
        fonts = [f.fname for f in matplotlib.font_manager.fontManager.ttflist if (('serif' in f.name.lower()) and not ('italic' in f.name.lower()))]
    return fonts[0]

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

def end_of_path(p, n=2):
    parts = p.parts
    p = Path(parts[-n])
    for i in range(-(n-1), 0):
        p/=parts[i]
    return p

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

# def is_confused_or_wrong(p,l,wrong_th=0.65):
#     for i in range(len(p)):
#         if ((abs(l[i].item() - p[i].item()) > wrong_th) or (p[i].item() >= 0.4 and p[i].item() <= 0.6)):
#             return True
#     return False

# def get_confused_or_wrong_names(preds,labels,names,wrong_th=0.65):
#     retrain = []
#     for i,p in enumerate(preds):
#         label,name = labels[i],names[i]
#         if is_confused_or_wrong(p,label,wrong_th):
#             retrain.append(name)
#     return retrain

# def is_confused(p):
#     for x in p:
#         if (x.item() >= 0.4 and x.item() <= 0.6):
#             return True
#     return False

# def is_wrong(p,l,wrong_th=0.65):
#     for x,y in zip(l,p):
#         if (abs(x.item() - y.item()) > wrong_th):
#             return True
#     return False

# def get_wrong(preds,labels,names,wrong_th=0.65):
#     retrain = []
#     for i,p in enumerate(preds):
#         label,name = labels[i],names[i]
#         if is_wrong(p,label,wrong_th):
#             retrain.append(name)
#     return retrain

# def get_confused(preds,labels,names):
#     retrain = []
#     for p,name in zip(preds,names):
#         if is_confused(p):
#             retrain.append(name)
#     return retrain

# def get_random_retrain(retrain,names):
#     for n in names:
#         if random.random() > 0.1 and n not in retrain:
#             retrain.append(n)
#     return retrain

# def get_retrain(model,loader,wrong_th=0.65):
#     retrain = []
#     for batch in loader:
#         preds,labels,names = torch.sigmoid(model.predict(batch[0])),batch[1],batch[2]
#         retrain += get_random_retrain(get_confused_or_wrong_names(preds,labels,names,wrong_th),names)
#     return retrain

# def get_wrong_class(preds,labels,names,class_idx,wrong_th=0.75):
#     retrain = []
#     for i,pred in enumerate(preds):
#         label,name = labels[i],names[i]
#         if ((label[class_idx].item() - pred[class_idx].item()) > wrong_th):
#             retrain.append(name)
#     return retrain

# def get_retrain_wrong_class(model,loader,class_idx,wrong_th=0.75):
#     retrain = []
#     for batch in loader:
#         preds,labels,names = torch.sigmoid(model.predict(batch[0])),batch[1],batch[2]
#         retrain += get_wrong_class(preds,labels,names,class_idx,wrong_th)
#     return retrain

# def get_confused_class(preds,names,class_idx):
#     retrain = []
#     for name,pred in zip(names,preds):
#         if pred[class_idx].item() >= 0.4 and pred[class_idx].item() <= 0.6:
#             retrain.append(name)
#     return retrain

# def get_retrain_confused_class(model,loader,class_idx):
#     retrain = []
#     for batch in loader:
#         preds,names = torch.sigmoid(model.predict(batch[0])),batch[2]
#         retrain += get_confused_class(preds,names,class_idx)
#     return retrain

# def get_all_zeros(preds,names):
#     retrain = []
#     for name,pred in zip(names,preds):
#         if sum(pred).item() == 0:
#             retrain.append(name)
#     return retrain

# def get_retrain_all_zeros(model,loader):
#     retrain = []
#     for batch in loader:
#         preds,names = torch.sigmoid(model.predict(batch[0])),batch[2]
#         retrain += get_all_zeros(preds,names)
#     return retrain

# def get_class(labels,names,class_idx):
#     retrain = []
#     for label,name in zip(labels,names):
#         if label[class_idx] == 1:
#             retrain.append(name)
#     return retrain

# def get_retrain_class(loader,class_idx):
#     retrain = []
#     for batch in loader:
#         labels,names = batch[1],batch[2]
#         retrain += get_class(labels,names,class_idx)
#     return retrain
