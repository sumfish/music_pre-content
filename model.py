import torch
import torch.nn as nn
import torch.nn.functional as F

class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, A, P, N):
        '''
        embedded_A, a = self.embeddingnet(A)
        embedded_P, p = self.embeddingnet(P)
        embedded_N, n = self.embeddingnet(N)
        '''
        embedded_A = self.embeddingnet(A)
        embedded_P = self.embeddingnet(P)
        embedded_N = self.embeddingnet(N)
        dist_AP = F.pairwise_distance(embedded_A, embedded_P, 2)
        dist_AN = F.pairwise_distance(embedded_A, embedded_N, 2)
        return dist_AP, dist_AN, embedded_A, embedded_P, embedded_N ###for training
        #return dist_AP, dist_AN, a, p, n ####for visualize

    def get_embedding(self, x):
        return self.embeddingnet(x)


class Net_s(nn.Module): ##internet
        def __init__(self):
            super(Net_s, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            print('level 0:{}'.format(x.shape))
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            print('level 1:{}'.format(x.shape))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            print('level 2:{}'.format(x.shape))
            x = x.view(-1, 320)
            print('level 3:{}'.format(x.shape))
            x = F.relu(self.fc1(x))
            print('level 4:{}'.format(x.shape))
            x = F.dropout(x, training=self.training)
            print('level 5:{}'.format(x.shape))
            return self.fc2(x)
#####################################################
class res_block(nn.Module):
    def __init__(self, inp, out, kernel, stride_list):
        super(res_block, self).__init__()
    
        self.conv1 = nn.Conv1d(inp, inp, kernel_size=kernel, stride=stride_list[0], padding=kernel//2)
        self.bn2 = nn.BatchNorm1d(inp)
        self.conv2 = nn.Conv1d(inp, out, kernel_size=kernel, stride=stride_list[1], padding=kernel//2)
        self.bn3 = nn.BatchNorm1d(out)
        self.add_conv = nn.Conv1d(inp, out, kernel_size=kernel, stride=stride_list[1], padding=kernel//2)
        if inp!=out:
            print('in!=out')
            downsample = True
        else:
            downsample = False
        self.downsample = downsample
        #self.up = nn.Conv1d(inp, out, kernel_size=kernel)
    
    def forward(self, x):
        '''
        block x:torch.Size([10, 128, 173])
        f(x):torch.Size([10, 128, 173])
        f(x):torch.Size([10, 128, 87])
        block x:torch.Size([10, 128, 87])
        '''
        #print('in')
        #print('block x:{}'.format(x.shape))  #shape(N,128,173)
        ori = x
        out = self.conv1(x) 
        out = F.relu(self.bn2(out))
        #print('f(x):{}'.format(out.shape))
        out = self.conv2(out)
        out = F.relu(self.bn3(out))
        #print('f(x):{}'.format(out.shape))
        if self.downsample:
            out = out + self.bn3(self.add_conv(ori))
            out = F.relu(out)
            #print('f(x):{}'.format(self.add_conv(ori).shape))
        else:
            out = out + F.avg_pool1d(ori, kernel_size=2, ceil_mode=True)
        #print('block x:{}'.format(out.shape))
        return out 


class Encoder_v4(nn.Module):    #new ,1d ,no connected layer
    def __init__(self):
        super(Encoder_v4, self).__init__()
        self.c_in=512 ###128=mel dimension
        self.c_out=1
        self.c_m1=128
        self.c_m2=64
        self.kernel=5
        self.stride=[1,2]
        self.conv1 = nn.Conv1d(self.c_in, self.c_m1, kernel_size=1) 
        self.norm_layer = nn.BatchNorm1d(self.c_m1)
        self.act = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25) 
        self.conv_last2 =res_block(self.c_m1, self.c_m2, self.kernel, self.stride)
        self.conv_last1 = res_block(self.c_m2, self.c_out, self.kernel, self.stride)  
        
        self.head = nn.Sequential(
            res_block(self.c_m1, self.c_m1, self.kernel, self.stride),
            res_block(self.c_m1, self.c_m1, self.kernel, self.stride),
            res_block(self.c_m1, self.c_m1, self.kernel, self.stride),
        )
        
        '''
        c_in=128
        self.head = nn.Sequential(
            res_block(self.c_in, self.c_in, self.kernel, self.stride),
            res_block(self.c_in, self.c_in, self.kernel, self.stride),
            res_block(self.c_in, self.c_m, self.kernel, self.stride),
            res_block(self.c_m, self.c_m, self.kernel, self.stride),
        )
        '''
    def forward(self, _input):
        x = _input
        #print('original:{}'.format(x.shape))
        x = x.view(-1, x.size(2), x.size(3))
        #print('after view:{}'.format(x.shape)) 
        
        #### conv bank??????

        #### dimension up
        x = self.conv1(x)
        x = self.norm_layer(x)
        x = self.act(x)
        x = self.drop_out(x)

        #### residual
        x = self.head(x)
        x = self.conv_last2(x)
        x = self.conv_last1(x)
        #print('level 1(after res):{}'.format(x.shape))
        x = x.view(-1, x.size(2))
        #print('level 1(after res):{}'.format(x.shape))
        #input()
        return x


class Encoder_v5(nn.Module):    #new ,1d ,no connected layer, 512
    def __init__(self):
        super(Encoder_v5, self).__init__()
        self.c_in=512 ###128=mel dimension
        self.c_out=1
        self.c_m1=128
        self.c_m2=64
        self.kernel=5
        self.stride=[1,2]
        self.conv1 = nn.Conv1d(self.c_in, self.c_in2, kernel_size=1) 
        self.norm_layer = nn.BatchNorm1d(self.c_in2)
        self.act = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)      

        self.head = nn.Sequential(
            res_block(self.c_in2, self.c_in2, self.kernel, self.stride),
            res_block(self.c_in2, self.c_in2, self.kernel, self.stride),
            res_block(self.c_in2, self.c_m, self.kernel, self.stride),
            res_block(self.c_m, self.c_m, self.kernel, self.stride),
            res_block(self.c_m, self.c_out, self.kernel, self.stride)
            #res_block(self.c_m, self.c_out, self.kernel, [1,1])#####new
        )
    def forward(self, _input):
        x = _input
        #print('original:{}'.format(x.shape))
        x = x.view(-1, x.size(2), x.size(3))
        #print('after view:{}'.format(x.shape)) 
        
        #### conv bank??????

        #### dimension up
        x = self.conv1(x)
        x = self.norm_layer(x)
        x = self.act(x)
        x = self.drop_out(x)

        #### residual
        x = self.head(x)
        #print('level 1(after res):{}'.format(x.shape))
        x = x.view(-1, x.size(2))
        #print('level 1(after res):{}'.format(x.shape))
        #input()
        return x

###################################################################
# frame-level paper
class block_in(nn.Module):
    def __init__(self, inp, out, kernel):
        super(block_in, self).__init__()
        if kernel==3:
            last_kernel=1
        else: 
            last_kernel=5
        self.in1 = nn.InstanceNorm2d(inp)
        self.conv1 = nn.Conv2d(inp, out, (kernel,1), padding=(1,0))
        self.in2 = nn.InstanceNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, (kernel,1), padding=(1,0))
        self.in3 = nn.InstanceNorm2d(out)
        self.up = nn.Conv2d(inp, out, (last_kernel,1), padding=(0,0))
        self.in4 = nn.InstanceNorm2d(out)
    
    def forward(self, x):
        #print('in')
        #print('block x:{}'.format(x.shape))  #shape(N,C,128,87)
        out = self.conv1(self.in1(x)) #before is a cnn layer
        #print('f(x):{}'.format(out.shape))
        out = self.conv2(F.relu(self.in2(out)))
        out = self.in3(out)
        #print('f(x):{}'.format(out.shape))
        #print('f(x):{}'.format(self.up(x).shape))
        out += self.in4(self.up(x)) ##########################
        #print('block x:{}'.format(out.shape))
        return out

class Encoder_v2(nn.Module):  ##add instance normalize 
    def __init__(self):
        super(Encoder_v2, self).__init__()
        fre = 64
        middle_size=50
        content_size=10
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(fre*3, middle_size)
        self.fc2 = nn.Linear(middle_size, content_size)
        #self.fc2 = nn.Linear(zsize, num_classes)
        self.lin_drop = nn.Dropout(p=0.5)
        
        self.head = nn.Sequential(
            #nn.BatchNorm2d(inp), ###############
            #nn.Conv2d(1, fre, (3,1), padding=(1,0)),
            nn.Conv2d(1, fre, (5,1), padding=(1,0)),
            block_in(fre, fre*2, 5),
            nn.Dropout(p=0.25),
            nn.MaxPool2d((3,1),(3,1)), #(42,T)
            
            block_in(fre*2, fre*3, 3),
            #nn.Dropout(p=0.3),
            ####nn.BatchNorm2d(fre*3),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d((3,1),(3,1)),
            #nn.Conv2d(fre*3, fre*2, (3,1), padding=(1,0))
        )
        

    def forward(self, _input):
        '''
        original:torch.Size([16, 1, 128, 87])
        level 1(after res):torch.Size([16, 192, 40, 87])
        level 2:torch.Size([16, 192, 1, 1])
        level 3:torch.Size([16, 192])
        level 4:torch.Size([16, 50])
        '''
        x = _input
        #print('original:{}'.format(x.shape))
        x = self.head(x)
        #print('level 1(after res):{}'.format(x.shape))
        x = self.avgpool(x)##############
        #print('level 2:{}'.format(x.shape))
        #x = x.view(-1, 192)
        x = torch.flatten(x, 1)
        #print('level 3:{}'.format(x.shape))
        last_layer = self.lin_drop(F.relu(self.fc1(x)))
        #print('level 4:{}'.format(x.shape))
        #out = F.softmax(self.fc2(last_layer), dim=0) ####classifier
        out = self.fc2(last_layer) 
        #print('level 5:{}'.format(x.shape))
        return out, last_layer

#############################################################################
class block_1d(nn.Module):
    def __init__(self, inp, out, kernel):
        super(block_1d, self).__init__()
        if kernel==3:
            last_kernel=1
        else: 
            last_kernel=5
        self.bn1 = nn.BatchNorm1d(inp)
        #self.conv1 = nn.Conv2d(inp, out, (kernel,1), padding=(1,0))
        self.conv1 = nn.Conv1d(inp, out, kernel_size=kernel, padding=1)
        self.bn2 = nn.BatchNorm1d(out)
        #self.conv2 = nn.Conv2d(out, out, (kernel,1), padding=(1,0))
        self.conv2 = nn.Conv1d(out, out, kernel_size=kernel, padding=1)
        self.bn3 = nn.BatchNorm1d(out)
        self.up = nn.Conv1d(inp, out, kernel_size=last_kernel)
        #self.up = nn.Conv2d(inp, out, (last_kernel,1), padding=(0,0))
    
    def forward(self, x):
        #print('in')
        #print('block x:{}'.format(x.shape))  #shape(N,C,128,87)
        out = self.conv1(self.bn1(x)) #before is a cnn layer
        #print('f(x):{}'.format(out.shape))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn3(out)
        #print('f(x):{}'.format(out.shape))
        #print('f(x):{}'.format(self.up(x).shape))
        out += self.up(x) ##########################
        #print('block x:{}'.format(out.shape))
        return out


class Encoder_v3(nn.Module):   ########## 1d conv
    def __init__(self):
        super(Encoder_v3, self).__init__()
        fre = 64
        msize=50
        zsize=10
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(fre*3, msize)
        self.fc2 = nn.Linear(msize, zsize)
        #self.fc2 = nn.Linear(zsize, num_classes)
        self.lin_drop = nn.Dropout(p=0.5)
        
        self.head = nn.Sequential(
            #nn.BatchNorm2d(inp), ###############
            #nn.Conv2d(1, fre, (3,1), padding=(1,0)),
            #nn.Conv2d(1, fre, (5,1), padding=(1,0)),
            nn.Conv1d(128, fre, 5),   #########  128=dictionary dimension
            block_1d(fre, fre*2, 5),
            nn.Dropout(p=0.25),
            #nn.MaxPool2d((3,1),(3,1)), #(42,T)
            
            block_1d(fre*2, fre*3, 3),
            #nn.Dropout(p=0.3),
            nn.BatchNorm1d(fre*3),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d((3,1),(3,1)),
            #nn.Conv2d(fre*3, fre*2, (3,1), padding=(1,0))
        )
        

    def forward(self, _input):
        x = _input
        #print('original:{}'.format(x.shape))
        
        x = x.view(-1, x.size(2), x.size(3))
        #print('after view:{}'.format(x.shape)) 
        x = self.head(x)
        #print('level 1(after res):{}'.format(x.shape))
        x = self.avgpool(x)##############
        #print('level 2:{}'.format(x.shape))
        #x = x.view(-1, 192)
        x = torch.flatten(x, 1)
        #print('level 3:{}'.format(x.shape))
        last_layer = self.lin_drop(F.relu(self.fc1(x)))
        #print('level 4:{}'.format(x.shape))
        out = self.fc2(last_layer) 
        #return out, last_layer
        return out
#############################################################################
# frame-level paper
class block(nn.Module):
    def __init__(self, inp, out, kernel):
        super(block, self).__init__()
        if kernel==3:
            last_kernel=1
        else: 
            last_kernel=5
        self.bn1 = nn.BatchNorm2d(inp)
        self.conv1 = nn.Conv2d(inp, out, (kernel,1), padding=(1,0))
        self.bn2 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, (kernel,1), padding=(1,0))
        self.bn3 = nn.BatchNorm2d(out)
        self.up = nn.Conv2d(inp, out, (last_kernel,1), padding=(0,0))
    
    def forward(self, x):
        #print('in')
        #print('block x:{}'.format(x.shape))  #shape(N,C,128,87)
        out = self.conv1(self.bn1(x)) #before is a cnn layer
        #print('f(x):{}'.format(out.shape))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn3(out)
        #print('f(x):{}'.format(out.shape))
        #print('f(x):{}'.format(self.up(x).shape))
        out += self.up(x) ##########################
        #print('block x:{}'.format(out.shape))
        return out


class Encoder_v1(nn.Module):
    def __init__(self):
        super(Encoder_v1, self).__init__()
        fre = 64
        msize=50
        zsize=10
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(fre*3, msize)
        self.fc2 = nn.Linear(msize, zsize)
        #self.fc2 = nn.Linear(zsize, num_classes)
        self.lin_drop = nn.Dropout(p=0.5)
        
        self.head = nn.Sequential(
            #nn.BatchNorm2d(inp), ###############
            #nn.Conv2d(1, fre, (3,1), padding=(1,0)),
            nn.Conv2d(1, fre, (5,1), padding=(1,0)),
            block(fre, fre*2, 5),
            nn.Dropout(p=0.25),
            nn.MaxPool2d((3,1),(3,1)), #(42,T)
            
            block(fre*2, fre*3, 3),
            #nn.Dropout(p=0.3),
            nn.BatchNorm2d(fre*3),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d((3,1),(3,1)),
            #nn.Conv2d(fre*3, fre*2, (3,1), padding=(1,0))
        )
        

    def forward(self, _input):
        '''
        original:torch.Size([16, 1, 128, 87])
        level 1(after res):torch.Size([16, 192, 40, 87])
        level 2:torch.Size([16, 192, 1, 1])
        level 3:torch.Size([16, 192])
        level 4:torch.Size([16, 50])
        '''
        x = _input
        #print('original:{}'.format(x.shape))
        x = self.head(x)
        #print('level 1(after res):{}'.format(x.shape))
        x = self.avgpool(x)##############
        #print('level 2:{}'.format(x.shape))
        #x = x.view(-1, 192)
        x = torch.flatten(x, 1)
        #print('level 3:{}'.format(x.shape))
        last_layer = self.lin_drop(F.relu(self.fc1(x)))
        #print('level 4:{}'.format(x.shape))
        #out = F.softmax(self.fc2(last_layer), dim=0) ####classifier
        out = self.fc2(last_layer) 
        #print('level 5:{}'.format(x.shape))
        return out, last_layer