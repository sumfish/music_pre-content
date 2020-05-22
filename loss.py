import numpy as np
import torch.nn.functional as F
import torch

def adaptive_weights_loss(dista, distb, margin, show=False):
    # softmax adaptive weights
    '''

    distance_neg: (distb)
    tensor([0.4764, 0.5987, 1.4089, 0.7002, 0.9701, 0.4005, 0.4819, 0.8708, 0.8494,
        1.1492], device='cuda:0', grad_fn=<NormBackward1>)

    softmax(-d_neg): (w_n)
    tensor([0.1310, 0.1159, 0.0515, 0.1047, 0.0799, 0.1413, 0.1303, 0.0883, 0.0902,
        0.0668], device='cuda:0', grad_fn=<SoftmaxBackward>)

    n_loss=w_n*distb:    
    tensor([0.0624, 0.0694, 0.0726, 0.0733, 0.0776, 0.0566, 0.0628, 0.0769, 0.0766,
        0.0768], device='cuda:0', grad_fn=<MulBackward0>)
    
    '''
    
    w_p=F.softmax(dista, dim=0)
    w_n=F.softmax(-distb, dim=0)
    
    furthest_positive=torch.sum(w_p*dista, dim=0)
    closest_negative=torch.sum(w_n*distb, dim=0)

    diff=furthest_positive-closest_negative
    loss=torch.max(diff+margin,torch.tensor(0.0).cuda())

    if (show==True):
        print('pos=%2f'%furthest_positive)
        print('neg=%2f'%closest_negative)
        print('pos-neg=%2f'%diff)
        print('loss=%3f'%loss)
    return loss