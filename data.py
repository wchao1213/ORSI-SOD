import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root,edge_root,back_gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.gts_back = [back_gt_root + f for f in os.listdir(back_gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)
        self.gts_back = sorted(self.gts_back)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.edge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        '''    
        self.edge2_transform = transforms.Compose([
            transforms.Resize(((self.trainsize)/4, (self.trainsize)/4)),
            transforms.ToTensor()])
        self.edge3_transform = transforms.Compose([
            transforms.Resize((self.trainsize/8, self.trainsize/8)),
            transforms.ToTensor()])
        self.edge4_transform = transforms.Compose([
            transforms.Resize((self.trainsize/16, self.trainsize/16)),
            transforms.ToTensor()])
        self.edge5_transform = transforms.Compose([
            transforms.Resize(((self.trainsize)/32, self.trainsize/32)),
            transforms.ToTensor()])
        '''
        self.gt_back_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        gt_back = self.binary_loader(self.gts_back[index])
        edge = self.binary_loader(self.edges[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        gt_back = self.gt_back_transform(gt_back)
        edge = self.edge_transform(edge)
        '''
        print('edge',edge)
        
        edge2 = self.edge2_transform(edge)
        print(edge2)
        edge3 = self.edge3_transform(edge)
        edge4 = self.edge4_transform(edge)
        edge5 = self.edge5_transform(edge)
        '''
        return image, gt, edge, gt_back#,edge2,edge3,edge4,edge5

    def filter_files(self):
        assert len(self.images) == len(self.gts) ==len(self.edges)  == len(self.gts_back)
        images = []
        gts = []
        edges = []
        gts_back = []
        for img_path, gt_path, edge_path,gt_back_path in zip(self.images, self.gts, self.edges, self.gts_back):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            edge = Image.open(edge_path)
            gt_back = Image.open(gt_back_path)
            if img.size == gt.size == edge.size == gt_back.size:
                images.append(img_path)
                gts.append(gt_path)
                edges.append(edge_path)
                gts_back.append(gt_back_path)
        self.images = images
        self.gts = gts
        self.edges = edges
        self.gts_back = gts_back

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt, edge, gt_back):
        assert img.size == gt.size == edge.size == gt_back
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), edge.resize((w, h), Image.NEAREST), gt_back.resize((w, h), Image.NEAREST)
        else:
            return img, gt, edge, gt_back

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, edge_root,gt_back_root, batchsize, trainsize, shuffle=False, num_workers=0, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, edge_root,gt_back_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, edge_root,gt_back_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts_back = [gt_back_root + f for f in os.listdir(gt_back_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)
        self.gts_back = sorted(self.gts_back)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.gt_back_transform = transforms.ToTensor()
        self.edge_transform =transforms.ToTensor()
        self.edge_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        gt_back = self.binary_loader(self.gts_back[self.index])
        edge = self.binary_loader(self.edges[self.index])
        #print('edge:',edge)
        edge = self.edge_transform(edge).unsqueeze(0)
        #print('edge:',edge)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, edge,gt_back, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


