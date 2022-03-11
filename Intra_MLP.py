import torch
import numpy

# codes of this function are borrowed from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
def index_points(device, points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    # batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    batch_indices = torch.arange(B, dtype=torch.long).cuda().view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def knn_l2(device, net, k, u):
    '''
    Input:
        k: int32, number of k in k-nn search
        net: (batch_size, npoint, c) float32 array, points
        u: int32, block size
    Output:
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    INF = 1e8
    batch_size = net.size(0)
    npoint = net.size(1)
    n_channel = net.size(2)

    square = torch.pow(torch.norm(net, dim=2,keepdim=True),2)

    def u_block(batch_size, npoint, u):
        block = numpy.zeros([batch_size, npoint, npoint])
        n = npoint // u
        for i in range(n):
            block[:, (i*u):(i*u+u), (i*u):(i*u+u)] = numpy.ones([batch_size, u, u]) * (-INF)
        return block

    # minus_distance = 2 * torch.matmul(net, net.transpose(2,1)) - square - square.transpose(2,1) + torch.Tensor(u_block(batch_size, npoint, u)).to(device)
    minus_distance = 2 * torch.matmul(net, net.transpose(2,1)) - square - square.transpose(2,1) + torch.Tensor(u_block(batch_size, npoint, u)).cuda()
    _, indices = torch.topk(minus_distance, k, largest=True, sorted=False)
    
    return indices

