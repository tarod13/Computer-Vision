import numpy as np
import pandas as pd
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
import torchvision.transforms as transforms

import argparse
from tensorboardX import SummaryWriter

# import multiprocessing
# multiprocessing.set_start_method('spawn', True)

ID = 0

ADAM_BETA1 = 0.9
USE_BIAS = 0
LR_DECAY = 1
LR_DECAY_FACTOR = 10
LR_DECAY_EPSD_1 = 120
LR_DECAY_EPSD_2 = 160

IMAGE_SIZE = 32
REPORT_LOSS_EVERY_EPOCH = 1
REPORT_ACC_EVERY_EPOCH = 1
PRINT_EVERY_EPOCH = 10
PRINT_EVERY_MBATCH = 100
SAVE_IMAGE_EVERY_EPOCH = 40

PATH_100 = "./data/cifar-100-python/meta"
PATH_10 = "./data/cifar-10-batches-py/batches.meta"

def unpickle(filepath):
    import pickle
    return pickle.load(open(filepath, 'rb'))  

def class_extractor(dataset):
    if dataset == 'CIFAR10':
        dict_classes = unpickle(PATH_10)
        return tuple(dict_classes['label_names'])
    else:
        dict_classes = unpickle(PATH_100)
        return tuple(dict_classes['fine_label_names'])


def weights_init_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)    

def weights_init_he(m):
     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0.0)  

def weights_init_unitary(m):
     if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, m.in_features**(-0.5), 0.02)
        torch.nn.init.constant_(m.bias, 0.0)

class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        
        stride = out_channels // in_channels
        self.non_linear_pipe = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=1, bias=USE_BIAS),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel, stride=1, padding=1, bias=USE_BIAS),
            nn.BatchNorm2d(out_channels),
        )
        
        if stride == 1:
            self.shortcut_pipe = nn.Sequential()
        else:
            self.shortcut_pipe = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )          
    
    def forward(self, x):
        Fx = self.non_linear_pipe(x)
        y = Fx + self.shortcut_pipe(x)
        Hx = F.relu(y)
        return Hx


class ResStack(nn.Module):
    def __init__(self, n_layers, n_filters, in_channels):
        super().__init__()
        
        layers = []
        in_channel_list = [in_channels] + (n_layers-1)*[n_filters]
        for id_layer in range(0, n_layers):
            layers.append(ResLayer(in_channel_list[id_layer], n_filters))
        self.layers = nn.Sequential(*layers) 
    
    def forward(self, x):
        return self.layers(x)     
    

class ResNet(nn.Module):
    def __init__(self, n_stacks, n_layers_per_stack, n_filters_0, out_dim=10):
        super().__init__()
        
        self.initial_pipe = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n_filters_0, kernel_size=3, stride=1, padding=1, bias=USE_BIAS),
            nn.BatchNorm2d(n_filters_0),
        )
        
        stacks = []
        for id_stack in range(0,n_stacks):
            in_channels = n_filters_0*(2**max(0,id_stack-1))
            n_filters = n_filters_0*(2**id_stack)
            stacks.append(ResStack(n_layers_per_stack, n_filters, in_channels))
        self.stack = nn.Sequential(*stacks)
        
        self.avg_pool_layer = nn.AvgPool2d(IMAGE_SIZE//(2**(n_stacks-1)))
        self.linear_layers = nn.Linear(n_filters_0*(2**(n_stacks-1)), out_dim)
    
    def forward(self, x):
        y = self.initial_pipe(x)
        y = self.stack(y)
        y = self.avg_pool_layer(y)
        y = self.linear_layers(y.view(y.size(0),-1))
        return y


class InstanceDiscNet(nn.Module):
    def __init__(self, n_instances, labels, n_classes, parametric, temperature, n_features, knn,
        n_stacks, n_layers_per_stack, n_filters_0):
        super().__init__()

        self.n_instances = n_instances
        self.n_classes = n_classes
        self.temperature = temperature
        self.parametric = parametric
        self.n_features = n_features
        self.knn = knn

        self.CNN_backbone = ResNet(n_stacks, n_layers_per_stack, n_filters_0, out_dim=self.n_features)
        self.labels = torch.LongTensor(labels).to(device)
        if not self.parametric:
            self.memory_bank = self.normalize(torch.randn(n_instances, self.n_features)).to(device)
            self.memory_bank.requires_grad = False
        else:
            self.recognition_layer = nn.Linear(self.n_features, n_instances)       
        
    @staticmethod
    def normalize(x):
        x += 1e-8
        norm = (x**2).sum(1, keepdim=True)**0.5
        return x / norm
    
    def forward(self, image):
        feature_vec = self.CNN_backbone(image)
        if not self.parametric:
            feature_vec = self.normalize(feature_vec)
        return feature_vec
    
    def non_parametric_logsoftmax(self, feature_vec):
        instance_class_logits_unnorm = torch.einsum('ij,nj->ni', self.memory_bank, feature_vec) / self.temperature
        log_norm_factor = torch.logsumexp(instance_class_logits_unnorm, dim=1, keepdim=True)
        instance_class_logits = instance_class_logits_unnorm - log_norm_factor
        return instance_class_logits
    
    def paramatric_logsoftmax(self, feature_vec):
        instance_class_logits_unnorm = self.recognition_layer(feature_vec)
        log_norm_factor = torch.logsumexp(instance_class_logits_unnorm, dim=1, keepdim=True)
        instance_class_logits = instance_class_logits_unnorm - log_norm_factor
        return instance_class_logits
    
    def logsoftmax(self, feature_vec):
        if self.parametric:
            return self.paramatric_logsoftmax(feature_vec)
        else:
            return self.non_parametric_logsoftmax(feature_vec)
    
    def update_prototypes(self, feature_vecs, instance_classes):
        if not self.parametric:
            self.memory_bank[instance_classes,:] = feature_vecs.detach().clone()
    
    def kNN_classification(self, test_example):
        with torch.no_grad():
            feature_vec = self(test_example)
            if self.parametric:
                instance_weights = torch.exp(self.recognition_layer(feature_vec))
            else:
                instance_weights = torch.exp(torch.einsum('ij,nj->ni', self.memory_bank, feature_vec) / self.temperature)
            knn_weights, knn_indices = torch.topk(instance_weights, self.knn)
            knn_labels = self.labels[knn_indices]
            one_hot_labels = torch.zeros(knn_labels.size(0)*self.knn, self.n_classes).to(device)
            one_hot_labels[np.arange(0,self.knn*knn_labels.size(0)),knn_labels.view(-1)] = torch.ones(knn_labels.size(0)*self.knn).to(device)
            voting = torch.einsum('ni,nic->nc', knn_weights, one_hot_labels.reshape(-1, self.knn, self.n_classes))            
            top5_classes = torch.topk(voting, 5)[1]
            top1_classes = torch.topk(voting, 1)[1]
            return top1_classes, top5_classes
    
    def init(self, method):
        self.apply(method)
        # if self.parametric:
        #     self.recognition_layer.apply(weights_init_unitary)


# Obtained from pytorch forum (cassidylaidlaw)
def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', default=False, action='store_true', help='Disable cuda computation')
    parser.add_argument('--load', default=False, action='store_true', help='Load previous net')
    parser.add_argument('--dataset', default='CIFAR10', action='store', help='Choose dataset used to train the model')
    parser.add_argument('--transform', default='Wu_augmentation', action='store', help='Choose method to transform images')
    parser.add_argument('--parametric', default=False, action='store_true', help='Choose parametric or non-parametric classification')
    parser.add_argument('--temp', default=0.07, action='store', help='Non-parametric temperature')
    parser.add_argument('--features', default=128, action='store', help='Number of latent features')
    parser.add_argument('--knn', default=200, action='store', help='Number of nearest neighbors')
    parser.add_argument('--optim', default='SGD', action='store', help='Choose optimizer')
    parser.add_argument('--lr', default=0.03, action='store', help='Learning rate for SGD optimization')
    parser.add_argument('--momentum', default=0.9, action='store', help='Momentum for SGD optimization')
    parser.add_argument('--weight_decay', default=5e-4, action='store', help='Weight decay for SGD optimization')
    parser.add_argument('--init', default='He', action='store', help='Choose initialization method')
    parser.add_argument('--epochs', default=200, action='store', help='Number of training epochs')
    parser.add_argument('--batch_size', default=128, action='store', help='Batch size for training')
    parser.add_argument('--stacks', default=4, action='store', help='(ResNet) Number of stacks with constant output size')
    parser.add_argument('--layers_per_stack', default=2, action='store', help='(ResNet) Number of conv layers between shortcuts')
    parser.add_argument('--init_filters', default=64, action='store', help='(ResNet) Number of filters in initial conv layer')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu else 'cuda')
    dataset = args.dataset if\
        args.dataset in ['CIFAR10', 'CIFAR100'] else 'CIFAR10'
    
    if args.transform == 'Basic':
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_transform = test_transform
    elif args.transform == 'Simple_augmentation':
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_transform = transforms.Compose(
            [transforms.RandomCrop(IMAGE_SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),            
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    train_load_set_args = {'root':'./data', 'train':True, 'download':True, 'transform':train_transform}
    test_load_set_args = {'root':'./data', 'train':False, 'download':True, 'transform':test_transform}
    if dataset == 'CIFAR10':
        CIFAR10_with_indices = dataset_with_indices(torchvision.datasets.CIFAR10)
        trainset = CIFAR10_with_indices(**train_load_set_args)
        testset = CIFAR10_with_indices(**test_load_set_args)
    else:
        CIFAR100_with_indices = dataset_with_indices(torchvision.datasets.CIFAR100)
        trainset = CIFAR100_with_indices(**train_load_set_args)
        testset = CIFAR100_with_indices(**test_load_set_args)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=2)

    classes = class_extractor(dataset)
    n_classes = len(classes)
    n_instances = len(trainset.train_labels)

    net = InstanceDiscNet(n_instances, trainset.train_labels, n_classes, 
                        args.parametric, args.temp, args.features, args.knn,
                        args.stacks, args.layers_per_stack, args.init_filters).to(device)
    if args.load:
        try:
            net.load_state_dict(torch.load('./nets/cifar_unfelnid'+str(ID)+'.pth'))
            net.train()            
        except:
            print("Previous ResNet wasn't found")
            pass
    else:
        if args.init == 'normal':
            net.init(weights_init_normal)
        elif args.init == 'he':
            net.init(weights_init_he)
        
    loss_function = nn.CrossEntropyLoss()
    learning_rate = args.lr
    if args.optim == 'Adam':
        optimizer = optim.Adam(
            params=net.parameters(), lr=learning_rate,
            betas=(ADAM_BETA1, 0.999)
        )
    elif args.optim == 'SGD':
        optimizer = optim.SGD(
            params=net.parameters(), lr=learning_rate, 
            momentum=args.momentum, weight_decay=args.weight_decay
            )
        
    writer = SummaryWriter()
    
    for epoch in range(0, args.epochs):
        avg_loss = 0.0
        for n_batch, data in enumerate(trainloader, 0):
            inputs, labels, indices = data[0].to(device), data[1].to(device), data[2].to(device).long()
            optimizer.zero_grad()
            feature_vecs = net(inputs)
            log_likelihoods = net.logsoftmax(feature_vecs)
            loss = loss_function(log_likelihoods, indices)
            loss.backward()
            optimizer.step()

            net.update_prototypes(feature_vecs.detach(), indices)

            avg_loss += (loss.item() - avg_loss)/(n_batch+1)
            if (n_batch+1) % PRINT_EVERY_MBATCH == 0:            
                print('[%d, %3d] loss: %.3f' % (epoch + 1, n_batch + 1, avg_loss), end="\r")            
        if (epoch+1) % PRINT_EVERY_EPOCH == 0:
            print('')

        if (epoch+1) % REPORT_LOSS_EVERY_EPOCH == 0:
            writer.add_scalar(
                'loss', avg_loss, epoch)

        if (epoch+1) % REPORT_ACC_EVERY_EPOCH == 0 or (epoch+1) == args.epochs:
            correct = 0
            total = 0        
            class_correct = list(0. for i in range(n_classes))
            class_total = list(0. for i in range(n_classes))
            confusion_matrix = np.zeros([n_classes,n_classes]).astype(int)
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    predicted = net.kNN_classification(images)[0].squeeze(1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    c = (predicted == labels).squeeze()
                    for i in range(len(labels)):
                        label = labels[i]
                        prediction = predicted[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1
                        confusion_matrix[label, prediction] += 1
            accuracy = 100*correct/total
            writer.add_scalar(
                'accuracy', accuracy, epoch)

            if epoch % SAVE_IMAGE_EVERY_EPOCH == 0:
                if dataset == 'CIFAR10':
                    confusion_df = pd.DataFrame(data=confusion_matrix, 
                                index=classes, columns=classes)

                    fig, ax = plt.subplots(figsize=(8,8))
                    hm = sns.heatmap(confusion_df, cmap="YlGnBu_r",
                                    annot=True, fmt="d", linewidths=.1, cbar=False)
                    hm.set_yticklabels(hm.get_yticklabels(), rotation = 0)
                else:
                    fig, ax = plt.subplots(figsize=(12,10))
                    hm = sns.heatmap(confusion_matrix, cmap="YlGnBu_r",
                                    annot=False, fmt="d", linewidths=0, cbar=True,
                                    xticklabels=False, yticklabels=False)
                
                writer.add_figure(
                    'confusion matrix', fig, epoch
                )
                plt.close()

        torch.save(net.state_dict(), './nets/cifar_unfelnid'+str(ID)+'.pth')

        if LR_DECAY and (((epoch+1) == LR_DECAY_EPSD_1) or ((epoch+1) == LR_DECAY_EPSD_2)):
            learning_rate = 0.1*learning_rate
            print('New learning rate: ' + str(learning_rate))
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
    
    writer.add_hparams({
        'lr': args.lr,
        'batch_size': args.batch_size,
        'n': args.layers_per_stack,
        'stacks': args.stacks,
        'epochs': args.epochs,
        'init_filters': n_filters_0,
        'transform': args.transform,
        'lr_decay': LR_DECAY,
        'weight decay': args.weight_decay,
        'use bias': USE_BIAS,
        'initialization': args.init,
        'optimizer': args.optim,
        'momentum': args.momentum,
        'knn': net.knn,
        'temperature': net.temperature,
        'n_features': net.n_features,
        'parametric': net.parametric
        },
        {'hparam/accuracy': accuracy, 'hparam/loss': avg_loss})