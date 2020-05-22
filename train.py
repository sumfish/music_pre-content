from __future__ import print_function
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from visdom import Visdom
import numpy as np
from dataloader import Content_Dataset
import datetime
from model import *
from utils import * 
from utils import __RandomCropNumpy
from evaluation import visual_tsne
from loss import adaptive_weights_loss
import yaml
from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='Content Classifier')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)') #10
parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=4, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--store_model_path', default='../checkpoint/')
parser.add_argument('--log_dir', default='./log')
parser.add_argument('--dataset_name', default='hung/2s_4class_512_no_normalize')
parser.add_argument('--model_name', default='model_v4_512_pic_w=128_add-blank_0520')
parser.add_argument('--name', default='adaptive_loss', type=str,
                    help='name of experiment')

####### setting
best_loss = 100
#model_name='model_v4(margin4)'
#model_name='base'
#dataset_name='/2s_have2track(16_17)'
#dataset_name='/2s_512'

def main():
    global args, best_loss, device, logger
    args = parser.parse_args()
    model_name=args.model_name
    dataset_name=args.dataset_name

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    '''
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    '''
    logger=Logger(args.log_dir)

    #global plotter 
    #plotter = VisdomLinePlotter(env_name=args.name)

    train_writer = SummaryWriter(
        os.path.join('.' + "/logs", 'hung_512_no_normalize(add_blank)'))
    # transform
    #read_data_minmax(dataset_name)
    #train_mean, train_std=compute_mean_std(dataset_name)
    audio_transform=transforms.Compose([
        #transforms.Lambda(lambda x: __RandomCropNumpy(x, 128)),  ##cut 128 frames
        transforms.ToTensor(),
    #    transforms.Normalize(mean=(train_mean,),std=(train_std,))
    ])

    # train/vali dataset
    train_data=Content_Dataset(dataset_name+'/train','data.npy', 'label.npy', 'triplet_blank.txt', transform = audio_transform)
    vali_split=0.1
    train_size = int((1-vali_split)* len(train_data))
    vali_size = len(train_data) - train_size
    train_dataset, vali_dataset = torch.utils.data.random_split(train_data, [train_size, vali_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    vali_loader = DataLoader(dataset=vali_dataset, batch_size=args.batch_size, shuffle=True)
    
    # test dataset
    #test_dataset = Content_Dataset('test/'+dataset_name+'/','data.npy', 'label.npy', 'triplet_s.txt', transform = audio_transform)
    test_dataset = Content_Dataset(dataset_name+'/test','data.npy', 'label.npy', 'triplet_blank.txt', transform = audio_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # model
    model=Encoder_v4()
    tnet = Tripletnet(model).to(device)

    # directory
    mkdir('../checkpoint/'+os.path.join(model_name,dataset_name))
    save_config(args)
    start_epoch=0
    save_dict={}
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            tnet.load_state_dict(checkpoint['state_dict'])
            
            param=tnet.state_dict()
            for k,v in param.items():
                print(k)
            
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    criterion = torch.nn.MarginRankingLoss(margin = args.margin)
    optimizer = optim.SGD(tnet.parameters(), lr=args.lr, momentum=args.momentum)

    
    # summary
    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))
    summary(model,(1,512,128))
    
    # draw 
    #vis = Visdom(env='no_normalize(content)')
    #### python -m visdom.server / http://localhost:8097
    #vis.line([[0.0, 0.0]], [0.] ,win='test', opts=dict(title='2s_Test', legend=['loss','acc']))
    #vis.line([[0.0, 0.0]], [0.] ,win='train', opts=dict(title= '2s_Train', legend=['loss','acc']))
    #vis.line([[0.0, 0.0]], [0.] ,win='vali', opts=dict(title= '2s_Vali', legend=['loss','acc']))

    # time
    print('Start training...')
    starttime = datetime.datetime.now()
    for epoch in range(start_epoch, args.epochs + 1):
        #### train for one epoch
        train_acc, train_loss=train(train_loader, tnet, criterion, optimizer, epoch)

        #### evaluate on validation/test set
        vali_acc, vali_loss= test(vali_loader, tnet, criterion, epoch, 'vali')
        test_acc, test_loss = test(test_loader, tnet, criterion, epoch, 'test')

        #### visualization for content
        #visual_tsne(tnet, device, test_loader, criterion, epoch, args.dataset_name)
        #input()
        
        #######
        train_writer.add_scalar('test/acc',test_acc,epoch)
        train_writer.add_scalar('test/loss',test_loss,epoch)
        train_writer.add_scalar('vali/acc',vali_acc,epoch)
        train_writer.add_scalar('vali/loss',vali_loss,epoch)
        #### remember best acc and save checkpoint
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)

        #if(epoch%3==0):
        if(is_best):
            print('preserve best checkpoint')
            save_dict['state_dict']=tnet.state_dict()
            save_dict['encoder']=model.state_dict()
            save_dict['epoch']=epoch+1
            save_dict['best_loss']=best_loss
            torch.save(save_dict,'../checkpoint/'+os.path.join(model_name,dataset_name)+'/content_%03d_en_best.pt'%(epoch))
        
        elif(epoch%3==0):
            print('preserve checkpoint')
            save_dict['state_dict']=tnet.state_dict()
            save_dict['encoder']=model.state_dict()
            save_dict['epoch']=epoch+1
            save_dict['best_loss']=best_loss
            torch.save(save_dict,'../checkpoint/'+os.path.join(model_name,dataset_name)+'/content_%03d_en.pt'%(epoch))
        '''    
        meta = {'train_loss': train_loss,
                'test_loss': test_loss,
                'test_acc': test_acc}
        logger.scalars_summary(args.model_name+args.dataset_name, meta, epoch)
        '''
        '''
        vis.line(X=torch.FloatTensor([epoch+1]), Y=torch.FloatTensor([train_loss]), win='train', update='append' if epoch+1 >0  else None,
            opts={'title': 'Train Loss'})
        vis.line(X=torch.FloatTensor([epoch+1]), Y=torch.FloatTensor([test_loss]), win='vali', update='append' if epoch+1 >0  else None,
            opts={'title': 'Test Loss'})
        vis.line(X=torch.FloatTensor([epoch+1]), Y=torch.FloatTensor([test_acc]), win='test', update='append' if epoch+1 >0  else None,
            opts={'title': 'Test Acc'})
        '''
        endtime = datetime.datetime.now()
        print (endtime - starttime)

def save_config(args):
        with open(f'{os.path.join(args.store_model_path,args.model_name,args.dataset_name)}/model.args.yaml', 'w') as f:
            #f.write(yaml.dump(vars(args), f))
            yaml.dump(vars(args), f)
        return

def train(train_loader, tnet, criterion, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()

    # switch to train mode
    tnet.train()
    show=False
    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        data1, data2, data3 = data1.to(device), data2.to(device), data3.to(device)
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        dista, distb, embedded_x, embedded_y, embedded_z = tnet(data1, data2, data3)
        #print('Distance a:{}, b:{}'.format(dista,distb))

        '''
        
        ###########################
        # 1 means, dista should be larger than distb(-1 => dista < distb)
        target = torch.FloatTensor(dista.size()).fill_(-1)
        target = target.to(device)
        target = Variable(target)
        loss_triplet = criterion(dista, distb, target)
        
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        
        loss = loss_triplet + 0.001 * loss_embedd
        #print(loss)
        #losses.update(loss_triplet.data.item(), data1.size(0))
        ###########################
        

        '''
        loss_triplet=adaptive_weights_loss(dista,distb,args.margin,show)
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd
        

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        losses.update(loss, data1.size(0))
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.data.item(), data1.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        show=False

        if batch_idx % args.log_interval == 0:
            show=True
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                losses.val, losses.avg, 
                100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))
    # log avg values to somewhere
    #plotter.plot('acc', 'train', epoch, accs.avg)
    #plotter.plot('loss', 'train', epoch, losses.avg)
    return accs.avg , losses.avg


def test(test_loader, tnet, criterion, epoch, mode):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluation mode
    tnet.eval()
    for batch_idx, (data1, data2, data3) in enumerate(test_loader):
        data1, data2, data3 = data1.to(device), data2.to(device), data3.to(device)
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        dista, distb, _, _, _ = tnet(data1, data2, data3)
        target = torch.FloatTensor(dista.size()).fill_(-1)
        target = target.to(device)
        target = Variable(target)
        test_loss =  criterion(dista, distb, target).data.item()

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        accs.update(acc, data1.size(0))
        losses.update(test_loss, data1.size(0))      

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    #plotter.plot('acc', mode, epoch, accs.avg)
    #plotter.plot('loss', mode, epoch, losses.avg)
    return accs.avg , losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, update='append', win=self.plots[var_name], name=split_name)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(dista, distb):
    margin = 0
    pred = (distb - dista - margin).cpu().data
    #print('Accuracy:{}'.format(float((pred > 0).sum())/dista.size()[0])) #correct num/batch size
    return float((pred > 0).sum())/dista.size()[0]

if __name__ == '__main__':
    main()    