from dreamai.utils import*

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
            img = np.array(self.data.iloc[index, 0])
        self.tfms = albu.Compose(self.transforms_)
        x = self.tfms(image=img)['image']
        if self.channels == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        return x,img_path
    
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