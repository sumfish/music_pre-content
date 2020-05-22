from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from model import *
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import *
from visualization import visualize_testing_data


config0={
    # path
    #'tsne_many':'../plots/tsne/2s/no_connected_spec128/adap_weight/margin4/',  ##change test vali name
    'tsne_many':'../plots/tsne/2s/no_connected_spec128/adap_weight/margin4/class4_hung_normalize/',
    'tsne':'../plots/tsne/2s/no_connected_spec512/adap_weight/point/',
    'sr':22050
}


def plot_with_labels(class_num, lowDWeights, labels, index_an, interval, epoch, raw=False, seg=5, if_many=False):
    #plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    print(lowDWeights.shape)
    
    ######### 
    start_seg=seg
    point=seg*3*class_num

    ##### apoint
    '''
    for i in range(7):
        c1 = cm.rainbow(int(255 * labels[point+3*i+1] / 7));
        plt.text(X[point+3*i+1], Y[point+3*i+1], 'P', backgroundcolor=c1, fontsize=7)
        print(labels[point+3*i+1])
        c2 = cm.rainbow(int(255 * labels[point+3*i+2] / 7));
        plt.text(X[point+3*i+2], Y[point+3*i+2], 'N', backgroundcolor=c2, fontsize=7)
        print(labels[point+3*i+2])

    x=X[point]
    y=Y[point]
    s=labels[point]
    c = cm.rainbow(int(255 * s / 7));
    print('Anchor class:{}'.format(s))
    plt.text(x, y, 'A', backgroundcolor=c, fontsize=7)
    '''
    
    ##### many
    marker=['o','v','s','*','+','D','^','x']
    for count in range(8):
        point=seg*3*class_num
        index_point=seg*class_num
        c1 = cm.rainbow(int(255 * count / 7))
        print('segment:{}'.format(seg))
        for i in range(class_num):
            ##### draw positive
            plt.plot(X[point+3*i+1], Y[point+3*i+1], color=c1, marker=marker[int(labels[point+3*i+1])], markersize=8)  #content->color, instrument category->shape
            #plt.plot(X[point+3*i+1], Y[point+3*i+1], color=c1, marker=marker[count], markersize=8)
            #c1 = cm.rainbow(int(255 * labels[point+3*i+1] / 7))
            #plt.plot(X[point+3*i+1], Y[point+3*i+1], color=c1, marker=marker[count], markersize=8) #content->shape, instrument category->color
            #plt.text(X[point+3*i+1], Y[point+3*i+1], str(count), backgroundcolor=c1, fontsize=7)
            print(labels[point+3*i+1]) ##instrument category
        x=X[point]
        y=Y[point]
        s=labels[point]
        print('anchor index:{}'.format(index_an[index_point])) #anchor index in .txt
        print('Anchor class:{}'.format(s))
        #plt.text(x, y, str(count), backgroundcolor=c, fontsize=7)
        #c = cm.rainbow(int(255 * s / 7))
        #plt.plot(x, y, color=c, marker=marker[count], markersize=8)
        c = cm.rainbow(int(255 * count / 7))
        plt.plot(x, y, color=c, marker=marker[int(labels[point])], markersize=8)
        
        seg=seg+interval

    plt.xlim(X.min(), X.max());
    plt.ylim(Y.min(), Y.max());
    plt.title('Visualize Content(4 class)')
    #plt.colorbar()
    print('Generate content img')
    if if_many:
        print(config0['tsne_many']+str(start_seg)+'_'+str(interval)+'/epoch%03d_7point.png'%(epoch))
        plt.savefig(config0['tsne_many']+str(start_seg)+'_'+str(interval)+'/epoch%03d_7point.png'%(epoch), format='png')
        #plt.savefig(path+'/epoch%03d_testest.png'%(epoch), format='png')
    else:
        print(config0['tsne']+'epoch%03d_%03d_anchor%d.png'%(epoch,seg,s))
        plt.savefig(config0['tsne']+'epoch%03d_%03d_anchor%d.png'%(epoch,seg,s), format='png')
    plt.close()
    
def predict(model, device, data_loader, criterion, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    all_labels=[]
    all_embs=[]
    indexs=[]

    with torch.no_grad():
        for batch_idx, (data1, data2, data3, label1, label2, label3, index) in enumerate(data_loader):
            data1, data2, data3 = data1.to(device), data2.to(device), data3.to(device)
            data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

            # compute output
            dista, distb, em_a, em_p, em_n = model(data1, data2, data3)
            target = torch.FloatTensor(dista.size()).fill_(-1)
            target = target.to(device)
            target = Variable(target)
            test_loss =  criterion(dista, distb, target).data.item()

            # measure accuracy and record loss
            acc = accuracy(dista, distb)
            accs.update(acc, data1.size(0))
            losses.update(test_loss, data1.size(0))      
            
            #visualization
            #### [A*batchsize] [P*batchsize] [N*batchsize] ->APNAPNAPNAPN
            if batch_idx==0: 
                indexs=index.cpu().data.numpy()
                for i in range(data1.size(0)):
                    if i==0:
                        all_embs=[em_a.cpu().data.numpy()[i]]
                    else:
                        all_embs=np.concatenate((all_embs,[em_a.cpu().data.numpy()[i]]), axis=0)
                    all_embs=np.concatenate((all_embs,[em_p.cpu().data.numpy()[i]]), axis=0)
                    all_embs=np.concatenate((all_embs,[em_n.cpu().data.numpy()[i]]), axis=0)
                    all_labels=np.concatenate((all_labels,[label1[i].numpy()]), axis=0)
                    all_labels=np.concatenate((all_labels,[label2[i].numpy()]), axis=0)
                    all_labels=np.concatenate((all_labels,[label3[i].numpy()]), axis=0)
            #elif batch_idx<=2:
            else:
                #all_embs=np.concatenate((all_embs,out_c.cpu().data.numpy()), axis=0)
                indexs=np.concatenate((indexs,index.cpu().data.numpy()), axis=0)

                for i in range(data1.size(0)):
                    all_embs=np.concatenate((all_embs,[em_a[i].cpu().data.numpy()]), axis=0)
                    all_embs=np.concatenate((all_embs,[em_p[i].cpu().data.numpy()]), axis=0)
                    all_embs=np.concatenate((all_embs,[em_n[i].cpu().data.numpy()]), axis=0)
                    all_labels=np.concatenate((all_labels,[label1[i].numpy()]), axis=0)
                    all_labels=np.concatenate((all_labels,[label2[i].numpy()]), axis=0)
                    all_labels=np.concatenate((all_labels,[label3[i].numpy()]), axis=0)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    
    return all_embs, all_labels, indexs

def visual_tsne(model, device, test_loader, c_criterion, epoch, datasetname):
    
    print('test_data:')
    num_seg=8
    class_num=4
    #segment=[50,200,500,700,900,1000] ### visualize point
    segment=[50,120,150,180,200,230]
    interval=[5,10,25,10,15,3]
    #segment=[200,500,600,700,900,1000] ### visualize point
    #interval=[10,25,15,15,15,15]
    if_many=True
    for i in range(len(segment)):
        if if_many:
            path=config0['tsne_many']+str(segment[i])+'_'+str(interval[i])
            mkdir(path)
        else:
            path=config0['tsne']
            mkdir(path)

    embs,labels, index_an =predict(model, device, test_loader, c_criterion, epoch)

    tsne =TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)  
    embs = tsne.fit_transform(embs) 
    #index_an = anchor index in .txt
    for i in range(len(segment)):
        visualize_testing_data(datasetname, segment[i],interval[i],num_seg,class_num-1,config0['tsne_many'])
        plot_with_labels(class_num-1, embs, labels, index_an, interval[i], epoch, raw=True, seg=segment[i], if_many=if_many) 
    


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
    #print('Accuracy:{}'.format(float((pred > 0).sum())/dista.size()[0]))
    return float((pred > 0).sum())/dista.size()[0]