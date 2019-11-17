import utils
import obj_utils
from dai_imports import*

class dai_image_csv_dataset(Dataset):
    
    def __init__(self, data_dir, data, transforms_ = None, obj = False, seg = False,
                    minorities = None, diffs = None, bal_tfms = None, channels = 3):
        super(dai_image_csv_dataset, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.transforms_ = transforms_
        self.tfms = None
        self.obj = obj
        self.seg = seg
        self.minorities = minorities
        self.diffs = diffs
        self.bal_tfms = bal_tfms
        self.channels = channels
        assert transforms_ is not None, print('Please pass some transforms.')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        if self.channels == 3:
            img = utils.bgr2rgb(cv2.imread(str(img_path)))
        else:    
            img = cv2.imread(str(img_path),0)

        # img = Image.open(img_path)
        # if self.channels == 3:
        #     img = img.convert('RGB')
        # else:    
        #     img = img.convert('L')

        y = self.data.iloc[index, 1]    
        if self.minorities and self.bal_tfms:
            if y in self.minorities:
                if hasattr(self.bal_tfms,'transforms'):
                    for tr in self.bal_tfms.transforms:
                        tr.p = self.diffs[y]
                    l = [self.bal_tfms]
                    l.extend(self.transforms_)
                    self.tfms = albu.Compose(l)    
                else:            
                    for t in self.bal_tfms:
                        t.p = self.diffs[y]
                    self.transforms_[1:1] = self.bal_tfms    
                    # self.tfms = transforms.Compose(self.transforms_)
                    self.tfms = albu.Compose(self.transforms_)
                    # print(self.tfms)
            else:
                # self.tfms = transforms.Compose(self.transforms_)
                self.tfms = albu.Compose(self.transforms_)
        else:    
            # self.tfms = transforms.Compose(self.transforms_)
            self.tfms = albu.Compose(self.transforms_)
        # x = self.tfms(img)
        x = self.tfms(image=img)['image']
        if self.channels == 1:
            x = x.unsqueeze(0)
        if self.seg:
            mask = Image.open(self.data.iloc[index, 1])
            seg_tfms = albu.Compose([self.tfms.transforms[0]])
            y = torch.from_numpy(np.array(seg_tfms(mask))).long().squeeze(0)

        # if self.obj:
        #     s = x.size()[1]
        #     if isinstance(s,tuple):
        #         s = s[0]
        #     row_scale = s/img.size[0]
        #     col_scale = s/img.size[1]
        #     y = rescale_bbox(y,row_scale,col_scale)
        #     y.squeeze_()
        #     y2 = self.data.iloc[index, 2]
        #     y = (y,y2)
        return (x,y,self.data.iloc[index, 0])

class dai_obj_dataset(Dataset):

    def __init__(self, data_dir, data, tfms, has_difficult=False):
        super(dai_obj_dataset, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.tfms = tfms
        self.has_difficult = has_difficult

        assert tfms is not None, print('Please pass some transforms.')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        image = Image.open(img_path)
        image = image.convert('RGB')
        try:
            boxes = torch.FloatTensor(literal_eval(self.data.iloc[index,1]))
            labels = torch.LongTensor(literal_eval(self.data.iloc[index,2]))
            if self.has_difficult:
                difficulties = torch.ByteTensor(literal_eval(self.data.iloc[index,3]))
            else:
                difficulties = None
        except:        
            boxes = torch.FloatTensor(self.data.iloc[index,1])
            labels = torch.LongTensor(self.data.iloc[index,2])
            if self.has_difficult:
                difficulties = torch.ByteTensor(self.data.iloc[index,3])
            else:
                difficulties = None

        # Apply transformations
        image, boxes, labels, difficulties = self.tfms(image, boxes, labels, difficulties)

        return image, boxes, labels, difficulties

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each

class dai_image_csv_dataset_food(Dataset):
    
    def __init__(self, data_dir, data, transforms_ = None):
        super(dai_image_csv_dataset_food, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.transforms_ = transforms_
        self.tfms = None
        assert transforms_ is not None, print('Please pass some transforms.')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        img = Image.open(img_path)
        img = img.convert('RGB')
        y1,y2 = self.data.iloc[index, 1],self.data.iloc[index, 2]    
        self.tfms = transforms.Compose(self.transforms_)    
        x = self.tfms(img)
        return (x,y1,y2)

class dai_image_csv_dataset_multi_head(Dataset):

    def __init__(self, data_dir, data, transforms_ = None, channels=3):
        super(dai_image_csv_dataset_multi_head, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.transforms_ = transforms_
        self.tfms = None
        self.channels = channels
        assert transforms_ is not None, print('Please pass some transforms.')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        if self.channels == 3:
            img = utils.bgr2rgb(cv2.imread(str(img_path)))
        else:    
            img = cv2.imread(str(img_path),0)

        # img = Image.open(img_path)
        # if self.channels == 3:
        #     img = img.convert('RGB')
        # else:    
        #     img = img.convert('L')

        y1,y2 = self.data.iloc[index, 1],self.data.iloc[index, 2]    
        self.tfms = albu.Compose(self.transforms_)    
        x = self.tfms(image=img)['image'].unsqueeze(0)
        # self.tfms = transforms.Compose(self.transforms_)    
        # x = self.tfms(img)
        return (x,y1,y2)

class dai_image_csv_dataset_landmarks(Dataset):

    def __init__(self, data_dir, data, transforms_ = None, obj = False,
                    minorities = None, diffs = None, bal_tfms = None,channels=3):
        super(dai_image_csv_dataset_landmarks, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.transforms_ = transforms_
        self.tfms = None
        self.obj = obj
        self.minorities = minorities
        self.diffs = diffs
        self.bal_tfms = bal_tfms
        self.channels = channels
        assert transforms_ is not None, print('Please pass some transforms.')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        img = Image.open(img_path)
        if self.channels == 3:
            img = img.convert('RGB')
        else:    
            img = img.convert('L')
        y1,y2 = self.data.iloc[index, 1],self.data.iloc[index, 2]
        try:
            y2 = torch.Tensor(literal_eval(y2))
        except:
            y2 = torch.Tensor(y2)
        self.tfms = transforms.Compose(self.transforms_)    
        x = self.tfms(img)
        s = x.shape[1]
        if isinstance(s,tuple):
            s = s[0]
        row_scale = s/img.size[1]
        col_scale = s/img.size[0]
        y2 = ((rescale_landmarks(copy.deepcopy(y2),row_scale,col_scale).squeeze())-s)/s
        return (x,y1,y2)

class dai_image_dataset(Dataset):

    def __init__(self, data_dir, data_df, input_transforms = None, target_transforms = None):
        super(dai_image_dataset, self).__init__()
        self.data_dir = data_dir
        self.data_df = data_df
        self.input_transforms = None
        self.target_transforms = None
        if input_transforms:
            self.input_transforms = transforms.Compose(input_transforms)
        if target_transforms:    
            self.target_transforms = transforms.Compose(target_transforms)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        img = utils.bgr2rgb(cv2.imread(str(img_path)))
        target = utils.bgr2rgb(cv2.imread(str(img_path)))
        if self.input_transforms:
            img = self.input_transforms(img)
        if self.target_transforms:
            target = self.target_transforms(target)
        return img, target

class dai_super_res_dataset(Dataset):

    def __init__(self, data_dir, data, transforms_,):
        super(dai_super_res_dataset, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.pre_transforms = albu.Compose(transforms_['pre_transforms'])
        self.input_transform = albu.Compose(transforms_['input'])
        self.target_transform = albu.Compose(transforms_['target'])
        self.resized_target_transform = albu.Compose(transforms_['resized_target'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        img_ = utils.bgr2rgb(cv2.imread(str(img_path)))
        print(self.pre_transforms.transforms.transforms)
        if len(self.pre_transforms.transforms.transforms) > 0:
            img_ = self.input_transform(image=img_)['image']    
        img = self.input_transform(image=img_)['image']
        target = self.target_transform(image=img_)['image']
        resized_target = self.resized_target_transform(image=img_)['image']
        return img, target, resized_target  

def rescale_landmarks(landmarks,row_scale,col_scale):
    landmarks2 = copy.deepcopy(torch.Tensor(landmarks).reshape((-1,2)))
    for lm in landmarks2:
        c,r = lm
        lm[0] = c*col_scale
        lm[1] = r*row_scale
        # lm[0] = c*row_scale
        # lm[1] = r*col_scale
    landmarks2 = landmarks2.reshape((1,-1))        
    return landmarks2

def listdir_fullpath(d):
        return [os.path.join(d, f) for f in os.listdir(d)]     

def get_minorities(df,thresh=0.8):

    c = df.iloc[:,1].value_counts()
    lc = list(c)
    max_count = lc[0]
    diffs = [1-(x/max_count) for x in lc]
    diffs = dict((k,v) for k,v in zip(c.keys(),diffs))
    minorities = [c.keys()[x] for x,y in enumerate(lc) if y < (thresh*max_count)]
    return minorities,diffs

def csv_from_path(path):

    path = Path(path)
    labels_paths = list(path.iterdir())
    tr_images = []
    tr_labels = []
    for l in labels_paths:
        if l.is_dir():
            for i in list(l.iterdir()):
                if i.suffix in IMG_EXTENSIONS:
                    name = i.name
                    label = l.name
                    new_name = f'{path.name}/{label}/{name}'
                    tr_images.append(new_name)
                    tr_labels.append(label)
    if len(tr_labels) == 0:
        return None
    tr_img_label = {'Img':tr_images, 'Label': tr_labels}
    csv = pd.DataFrame(tr_img_label,columns=['Img','Label'])
    csv = csv.sample(frac=1).reset_index(drop=True)
    return csv

def add_extension(a,e):
    a = [x+e for x in a]
    return a

def one_hot(targets, multi = False):
    if multi:
        binerizer = MultiLabelBinarizer()
        dai_1hot = binerizer.fit_transform(targets)
    else:
        binerizer = LabelBinarizer()
        dai_1hot = binerizer.fit_transform(targets)
    return dai_1hot,binerizer.classes_

def get_img_stats(dataset,channels):

    print('Calculating mean and std of the data for standardization. Might take some time, depending on the training data size.')

    imgs = []
    for d in dataset:
        img = d[0]
        imgs.append(img)
    imgs_ = torch.stack(imgs,dim=3)
    imgs_ = imgs_.view(channels,-1)
    imgs_mean = imgs_.mean(dim=1)
    imgs_std = imgs_.std(dim=1)
    del imgs
    del imgs_
    print('Done')
    return imgs_mean,imgs_std

def split_df(train_df,test_size = 0.15):
    try:    
        train_df,val_df = train_test_split(train_df,test_size = test_size,random_state = 2,stratify = train_df.iloc[:,1])
    except:
        train_df,val_df = train_test_split(train_df,test_size = test_size,random_state = 2)
    train_df = train_df.reset_index(drop = True)
    val_df =  val_df.reset_index(drop = True)
    return train_df,val_df    

def save_obj(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class DataProcessor:
    
    def __init__(self, data_path = None, train_csv = None, val_csv = None,test_csv = None, class_names = [], seg = False, obj = False, sr = False,
                 multi_label=False,multi_head = False,tr_name = 'train', val_name = 'val', test_name = 'test', extension = None, setup_data = True):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        (self.data_path,self.train_csv,self.val_csv,self.test_csv,
         self.tr_name,self.val_name,self.test_name,self.extension) = (data_path,train_csv,val_csv,test_csv,
                                                                      tr_name,val_name,test_name,extension)
        self.seg = seg
        self.obj = obj
        self.sr = sr
        self.multi_head = multi_head
        self.multi_label = multi_label
        self.single_label = False
        self.img_mean = self.img_std = None
        self.data_dir,self.num_classes,self.class_names = data_path,len(class_names),class_names
        if setup_data:
            self.set_up_data()
                
    def set_up_data(self,split_size = 0.15):

        (data_path,train_csv,val_csv,test_csv,tr_name,val_name,test_name) = (self.data_path,self.train_csv,self.val_csv,self.test_csv,
                                                                             self.tr_name,self.val_name,self.test_name)

        # check if paths given and also set paths
        
        if not data_path:
            data_path = os.getcwd() + '/'
            self.data_dir = data_path
        tr_path = os.path.join(data_path,tr_name)
        val_path = os.path.join(data_path,val_name)
        test_path = os.path.join(data_path,test_name)
        os.makedirs('mlflow_saved_training_models',exist_ok=True)
        if train_csv is None:
            # if (os.path.exists(os.path.join(data_path,tr_name+'.csv'))):
            #     train_csv = tr_name+'.csv'
            #     if os.path.exists(os.path.join(data_path,val_name+'.csv')):
            #         val_csv = val_name+'.csv'
            #     if os.path.exists(os.path.join(data_path,test_name+'.csv')):
            #         test_csv = test_name+'.csv'
            # else:
            train_csv,val_csv,test_csv = self.data_from_paths_to_csv(data_path,tr_path,val_path,test_path)
        else:
            self.data_dir = tr_path

        train_csv_path = os.path.join(data_path,train_csv)
        train_df = pd.read_csv(train_csv_path)
        if 'Unnamed: 0' in train_df.columns:
            train_df = train_df.drop('Unnamed: 0', 1)
        img_names = [str(x) for x in list(train_df.iloc[:,0])]
        if self.extension:
            img_names = add_extension(img_names,self.extension)
        if val_csv is not None:
            val_csv_path = os.path.join(data_path,val_csv)
            val_df = pd.read_csv(val_csv_path)
            val_targets = list(val_df.iloc[:,1].apply(lambda x: str(x)))
        if test_csv is not None:
            test_csv_path = os.path.join(data_path,test_csv)
            test_df = pd.read_csv(test_csv_path)
            test_targets = list(test_df.iloc[:,1].apply(lambda x: str(x)))
        if self.seg:
            print('\nSemantic Segmentation\n')
        elif self.obj:
            print('\nObject Detection\n')
        elif self.sr:
            print('\nSuper Resolution\n')
        else:
            if self.multi_head:
                print('\nMulti-head Classification\n')

                train_df.fillna('',inplace=True)
                train_df_single = train_df[[train_df.columns[0],train_df.columns[1]]].copy() 
                train_df_multi = train_df[[train_df.columns[0],train_df.columns[2]]].copy()
                
                targets = list(train_df_multi.iloc[:,1].apply(lambda x: str(x)))
                lengths = [len(t) for t in [s.split() for s in targets]]
                split_targets = [t.split() for t in targets]
                try:
                    split_targets = [list(map(int,x)) for x in split_targets]
                except:
                    pass
                dai_onehot,onehot_classes = one_hot(split_targets,multi=True)
                train_df_multi.iloc[:,1] = [torch.from_numpy(x).type(torch.FloatTensor) for x in dai_onehot]
                self.num_multi_classes,self.multi_class_names = len(onehot_classes),onehot_classes

                targets = list(train_df_single.iloc[:,1].apply(lambda x: str(x)))
                lengths = [len(t) for t in [s.split() for s in targets]]
                split_targets = [t.split() for t in targets]
                unique_targets = list(np.unique(targets))
                try:
                    unique_targets.sort(key=int)
                except:
                    unique_targets.sort()
                unique_targets_dict = {k:v for v,k in enumerate(unique_targets)}
                train_df_single.iloc[:,1] = pd.Series(targets).apply(lambda x: unique_targets_dict[x])
                self.num_classes,self.class_names = len(unique_targets),unique_targets

                train_df = pd.merge(train_df_single,train_df_multi,on=train_df_single.columns[0])

            elif self.multi_label:
                print('\nMulti-label Classification\n')

                train_df_concat = train_df.copy()
                if val_csv:
                    train_df_concat  = pd.concat([train_df_concat,val_df]).reset_index(drop=True,inplace=False)
                if test_csv:
                    train_df_concat  = pd.concat([train_df_concat,test_df]).reset_index(drop=True,inplace=False)

                train_df_concat.fillna('',inplace=True)
                targets = list(train_df_concat.iloc[:,1].apply(lambda x: str(x)))
                lengths = [len(t) for t in [s.split() for s in targets]]
                split_targets = [t.split() for t in targets]
                try:
                    split_targets = [list(map(int,x)) for x in split_targets]
                except:
                    pass
                dai_onehot,onehot_classes = one_hot(split_targets,self.multi_label)
                train_df_concat.iloc[:,1] = [torch.from_numpy(x).type(torch.FloatTensor) for x in dai_onehot]
                train_df = train_df_concat.loc[:len(train_df)-1].copy()
                if val_csv:
                    val_df = train_df_concat.loc[len(train_df):len(train_df)+len(val_df)-1].copy().reset_index(drop=True)
                if test_csv:
                    test_df = train_df_concat.loc[len(val_df)+len(train_df):len(val_df)+len(train_df)+len(test_df)-1].copy().reset_index(drop=True)
                self.num_classes,self.class_names = len(onehot_classes),onehot_classes

                # train_df.fillna('',inplace=True)
                # targets = list(train_df.iloc[:,1].apply(lambda x: str(x)))
                # lengths = [len(t) for t in [s.split() for s in targets]]
                # split_targets = [t.split() for t in targets]
                # try:
                #     split_targets = [list(map(int,x)) for x in split_targets]
                # except:
                #     pass
                # dai_onehot,onehot_classes = one_hot(split_targets,self.multi_label)
                # train_df.iloc[:,1] = [torch.from_numpy(x).type(torch.FloatTensor) for x in dai_onehot]

            else:
                print('\nSingle-label Classification\n')

                targets = list(train_df.iloc[:,1].apply(lambda x: str(x)))
                lengths = [len(t) for t in [s.split() for s in targets]]
                split_targets = [t.split() for t in targets]                
                self.single_label = True
                unique_targets = list(np.unique(targets))
                try:
                    unique_targets.sort(key=int)
                except:
                    unique_targets.sort()
                unique_targets_dict = {k:v for v,k in enumerate(unique_targets)}
                train_df.iloc[:,1] = pd.Series(targets).apply(lambda x: unique_targets_dict[x])
                if val_csv:
                    val_df.iloc[:,1] = pd.Series(val_targets).apply(lambda x: unique_targets_dict[x])
                if test_csv:
                    test_df.iloc[:,1] = pd.Series(test_targets).apply(lambda x: unique_targets_dict[x])   
                self.num_classes,self.class_names = len(unique_targets),unique_targets

        if not val_csv:
            train_df,val_df = split_df(train_df,split_size)
        if not test_csv:    
            val_df,test_df = split_df(val_df,split_size)
        tr_images = [str(x) for x in list(train_df.iloc[:,0])]
        val_images = [str(x) for x in list(val_df.iloc[:,0])]
        test_images = [str(x) for x in list(test_df.iloc[:,0])]
        if self.extension:
            tr_images = add_extension(tr_images,self.extension)
            val_images = add_extension(val_images,self.extension)
            test_images = add_extension(test_images,self.extension)
        train_df.iloc[:,0] = tr_images
        val_df.iloc[:,0] = val_images
        test_df.iloc[:,0] = test_images
        if self.single_label:
            dai_df = pd.concat([train_df,val_df,test_df]).reset_index(drop=True,inplace=False)
            dai_df.iloc[:,1] = [self.class_names[x] for x in dai_df.iloc[:,1]]
            # train_df.iloc[:,1] = [self.class_names[x] for x in train_df.iloc[:,1]]
            # val_df.iloc[:,1] = [self.class_names[x] for x in val_df.iloc[:,1]]
            # test_df.iloc[:,1] = [self.class_names[x] for x in test_df.iloc[:,1]]
            dai_df.to_csv(os.path.join(data_path,'dai_processed_df.csv'),index=False)
        train_df.to_csv(os.path.join(data_path,'dai_{}.csv'.format(self.tr_name)),index=False)
        val_df.to_csv(os.path.join(data_path,'dai_{}.csv'.format(self.val_name)),index=False)
        test_df.to_csv(os.path.join(data_path,'dai_{}.csv'.format(self.test_name)),index=False)
        self.minorities,self.class_diffs = None,None
        if self.single_label:
            self.minorities,self.class_diffs = get_minorities(train_df)
        self.data_dfs = {self.tr_name:train_df, self.val_name:val_df, self.test_name:test_df}
        data_dict = {'data_dfs':self.data_dfs,'data_dir':self.data_dir,'num_classes':self.num_classes,'class_names':self.class_names,
                    # 'num_multi_classes':self.num_multi_classes,'multi_class_names':self.multi_class_names,
                    'minorities':self.minorities,'class_diffs':self.class_diffs,'seg':self.seg,'obj':self.obj,'sr':self.sr,
                    'single_label':self.single_label,'multi_label':self.multi_label}
        self.data_dict = data_dict
        return data_dict

    def data_from_paths_to_csv(self,data_path,tr_path,val_path = None,test_path = None):
            
        train_df = csv_from_path(tr_path)
        train_df.to_csv(os.path.join(data_path,f'dai_{self.tr_name}.csv'),index=False)
        ret = (f'dai_{self.tr_name}.csv',None,None)
        if val_path is not None:
            if os.path.exists(val_path):
                val_df = csv_from_path(val_path)
                if val_df is not None:
                    val_df.to_csv(os.path.join(data_path,f'dai_{self.val_name}.csv'),index=False)
                    ret = (f'dai_{self.tr_name}.csv',f'dai_{self.val_name}.csv',None)
        if test_path is not None:
            if os.path.exists(test_path):
                test_df = csv_from_path(test_path)
                if test_df is not None:
                    test_df.to_csv(os.path.join(data_path,f'dai_{self.test_name}.csv'),index=False)
                    ret = (f'dai_{self.tr_name}.csv',f'dai_{self.val_name}.csv',f'dai_{self.test_name}.csv')        
        return ret
        
    def get_data(self, data_dict = None, s = (224,224), dataset = dai_image_csv_dataset, train_resize_transform = None, val_resize_transform = None, 
                 bs = 32, balance = False, super_res_crop = 256, super_res_upscale_factor = 1,
                 tfms = [],bal_tfms = None,num_workers = 8, stats_percentage = 0.6,channels = 3, normalise = True, img_mean = None, img_std = None):
        
        self.image_size = s
        if not data_dict:
            data_dict = self.data_dict
        data_dfs,data_dir,minorities,class_diffs,single_label,seg,obj,sr = (data_dict['data_dfs'],data_dict['data_dir'],
                                                        data_dict['minorities'],data_dict['class_diffs'],
                                                        data_dict['single_label'],data_dict['seg'],data_dict['obj'],data_dict['sr'])
        if not single_label:
           balance = False                                                 
        if not bal_tfms:
            bal_tfms = { self.tr_name: [albu.HorizontalFlip()],
                         self.val_name: None,
                         self.test_name: None 
                       }
        else:
            bal_tfms = {self.tr_name: bal_tfms, self.val_name: None, self.test_name: None}

        # resize_transform = transforms.Resize(s,interpolation=Image.NEAREST)
        if train_resize_transform is None:
            train_resize_transform = albu.Resize(s[0],s[1],interpolation=2)          
        if img_mean is None and self.img_mean is None: # and not sr:
            # temp_tfms = [resize_transform, transforms.ToTensor()]
            temp_tfms = [train_resize_transform, AT.ToTensor()]
            frac_data = data_dfs[self.tr_name].sample(frac = stats_percentage).reset_index(drop=True).copy()
            temp_dataset = dai_image_csv_dataset(data_dir = data_dir,data = frac_data,transforms_ = temp_tfms,channels = channels)
            self.img_mean,self.img_std = get_img_stats(temp_dataset,channels)
        elif self.img_mean is None:
            self.img_mean,self.img_std = img_mean,img_std
        # if obj:
        #     obj_transform = obj_utils.transform(size = s, mean = self.img_mean, std = self.img_std)
        #     dataset = dai_obj_dataset
        #     tfms = obj_transform.train_transform
        #     val_test_tfms = obj_transform.val_test_transform
        #     data_transforms = {
        #         self.tr_name: tfms,
        #         self.val_name: val_test_tfms,
        #         self.test_name: val_test_tfms
        #     }
        #     has_difficult = (len(data_dfs[self.tr_name].columns) == 4)
        #     image_datasets = {x: dataset(data_dir = data_dir,data = data_dfs[x],tfms = data_transforms[x],
        #                         has_difficult = has_difficult)
        #                       for x in [self.tr_name, self.val_name, self.test_name]}
        #     dataloaders = {x: DataLoader(image_datasets[x], batch_size=bs,collate_fn=image_datasets[x].collate_fn,
        #                                                 shuffle=True, num_workers=num_workers)
        #                 for x in [self.tr_name, self.val_name, self.test_name]}
        if sr:
            super_res_crop = super_res_crop - (super_res_crop % super_res_upscale_factor)
            super_res_transforms = {
                'pre_transforms':[albu.RandomCrop(super_res_crop,super_res_crop)]+tfms,
                'input':[
                        # albu.CenterCrop(super_res_crop,super_res_crop),
                        albu.Resize((super_res_crop // super_res_upscale_factor),
                                    (super_res_crop // super_res_upscale_factor),
                                    interpolation = 2),
                        albu.Normalize(self.img_mean,self.img_std),
                        AT.ToTensor()
                ],
                'target':[
                        AT.ToTensor()
                ],
                'resized_target':[
                    albu.Resize((super_res_crop // super_res_upscale_factor),
                                (super_res_crop // super_res_upscale_factor),
                                interpolation = 2),
                    albu.Resize(super_res_crop,super_res_crop,interpolation = 2),
                    AT.ToTensor()
                ]
            }
            image_datasets = {x: dataset(data_dir,data_dfs[x],super_res_transforms)
                             for x in [self.tr_name, self.val_name, self.test_name]}

        else:   
            if len(tfms) == 0:
                # if normalise:
                #     tfms = [
                #         resize_transform,
                #         transforms.ToTensor(),
                #         transforms.Normalize(normalise_array,normalise_array)
                #     ]
                # else:
                #     tfms = [
                #         resize_transform,
                #         transforms.ToTensor()
                #     ]    
                tfms = [
                        train_resize_transform,
                        # transforms.ToTensor(),
                        # transforms.Normalize(self.img_mean,self.img_std)
                        albu.Normalize(self.img_mean,self.img_std),
                        AT.ToTensor()
                    ]
            else:
                tfms_temp = [
                    train_resize_transform,
                    # transforms.ToTensor(),
                    # transforms.Normalize(self.img_mean,self.img_std)
                    albu.Normalize(self.img_mean,self.img_std),
                    AT.ToTensor()
                ]
                tfms_temp[1:1] = tfms
                tfms = tfms_temp
                print('Transforms: ',)
                print(tfms)
                print()
            if val_resize_transform is None:
                val_resize_transform = albu.Resize(s[0],s[1],interpolation=2)
            val_test_tfms = [
                val_resize_transform,
                # transforms.ToTensor(),
                # transforms.Normalize(self.img_mean,self.img_std)
                albu.Normalize(self.img_mean,self.img_std),
                AT.ToTensor()
            ]
            data_transforms = {
                self.tr_name: tfms,
                self.val_name: val_test_tfms,
                self.test_name: val_test_tfms
            }

            # if balance:
            #     image_datasets = {x: dataset(data_dir = data_dir,data = data_dfs[x],
            #                                 transforms_ = data_transforms[x],minorities = minorities,diffs = class_diffs,
            #                                 bal_tfms = bal_tfms[x],channels = channels,seg = seg)
            #                 for x in [self.tr_name, self.val_name, self.test_name]}    
            if self.multi_head:
                dataset = dai_image_csv_dataset_multi_head
                image_datasets = {x: dataset(data_dir = data_dir,data = data_dfs[x],
                                            transforms_ = data_transforms[x],channels = channels)
                            for x in [self.tr_name, self.val_name, self.test_name]}
            else:
                image_datasets = {x: dataset(data_dir = data_dir,data = data_dfs[x],
                                            transforms_ = data_transforms[x],channels = channels,seg = seg)
                            for x in [self.tr_name, self.val_name, self.test_name]}
        
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=bs,
                                                    shuffle=True, num_workers=num_workers)
                    for x in [self.tr_name, self.val_name, self.test_name]}
        dataset_sizes = {x: len(image_datasets[x]) for x in [self.tr_name, self.val_name, self.test_name]}
        
        self.image_datasets,self.dataloaders,self.dataset_sizes = (image_datasets,dataloaders,dataset_sizes)
        
        return image_datasets,dataloaders,dataset_sizes