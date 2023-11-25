import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from torch.nn import Parameter
from IPython import embed
import numpy as np
MIN_NUM_PATCHES = 15
# from vit_pytorch_my.mobilenet_v3_my import MobileNetV3_backbone
from vit_pytorch_my.mobilenet_v3_my import MobileNetV3_backbone
# from vit_pytorch_my.EfficientNet import EfficientNet
# from vit_pytorch_my.mobilevit import MobileViT
import pdb
from timm.models.layers import DropPath
# import face_alignment
# from vit_pytorch_my import FAN_network
import torchvision.models as models
import math
import os
import logging
from torch.nn.functional import normalize, linear
import torch.distributed as dist

from torch.autograd import Variable
# from vit_pytorch_my import ada_losses,spsher_loss
# import dgl
import time
class CosFace(nn.Module):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        s: norm of input feature
        m: margin
        cos(theta)-m
    """

    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.4):#0.35
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.m = m
        print("self.device_id", self.device_id)
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------

        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])),
                                   dim=1)
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        # pdb.set_trace()
        if len(label.shape)>1:
            if self.device_id == None:
                one_hot=label.cuda()
            else:
                one_hot=label.cuda(self.device_id[0])
        else:
            if self.device_id != None:
                one_hot = one_hot.cuda(self.device_id[0])
            # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot

                one_hot.scatter_(1, label.cuda(self.device_id[0]).view(-1, 1).long(), 1)
            else:
                in_device=label.device
                one_hot=one_hot.to(in_device)
                one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'

class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """

    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.50, easy_margin=False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])),
                                   dim=1)
        cosine=cosine.float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi=phi.float()
        # pdb.set_trace()
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # pdb.set_trace()
        if len(label.shape)>1:
            if self.device_id == None:
                one_hot=label.cuda()
            else:
                one_hot=label.cuda(self.device_id[0])
        else:
            one_hot = torch.zeros(cosine.size())
            if self.device_id != None:
                one_hot = one_hot.cuda(self.device_id[0])
            else:
                in_device=label.device
                one_hot=one_hot.to(in_device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output


class AdaFace(nn.Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1,
                 ):
        super(AdaFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        print('\n\AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def forward(self, embbedings, label):
        # norm = torch.norm(embbedings, 2, 1, True)
        # norms = torch.div(embbedings, norm)
        norms = torch.norm(embbedings, 2, 1, True)
        embbedings = torch.div(embbedings, norms)
        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # g_angular
        # pdb.set_trace()
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
class Residual_droppath(nn.Module):
    def __init__(self, fn,drop_path_rate=0.1):
        super().__init__()
        self.fn = fn
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
    def forward(self, x, **kwargs):
        return self.drop_path(self.fn(x, **kwargs)) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.,last_dim=None):
        super().__init__()
        if last_dim==None:
            last_dim=dim
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, last_dim),
            nn.Dropout(dropout)
        )
        # else:
            
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        # self.to_qkv = nn.Linear(dim, np.int64(inner_dim/2) , bias = False)
        # self.to_out = nn.Sequential(
        #     nn.Linear(np.int64(inner_dim/6), dim),
        #     nn.Dropout(dropout)
        # )
        self.attention_score = 0

    def forward(self, x, mask = None):
        # pdb.set_trace()
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        #embed()
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        # pdb.set_trace()
        self.attention_score=attn.detach()
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        return out

class Transformer(nn.Module):
    # def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
    #     super().__init__()
    #     self.layers = nn.ModuleList([])
    #     for _ in range(depth):
    #         self.layers.append(nn.ModuleList([
    #             Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
    #             Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
    #         ]))
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout,last_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        if last_dim==None:
            last_dim=dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual_droppath(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual_droppath(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
        # pdb.set_trace()
        # self.layers.append(nn.ModuleList([
        #         Residual_droppath(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
        #         Residual_droppath(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout,last_dim=last_dim)))
        #     ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            #embed()
            x = ff(x)
        return x



# class ArcFace(nn.Module):
#     r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
#         Args:
#             in_features: size of each input sample
#             out_features: size of each output sample
#             device_id: the ID of GPU where the model will be trained by model parallel.
#                        if device_id=None, it will be trained on CPU without model parallel.
#             s: norm of input feature
#             m: margin
#             cos(theta+m)
#         """

#     def __init__(self, in_features, out_features, device_id, s=64.0, m=0.50, easy_margin=False):
#         super(ArcFace, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.device_id = device_id

#         self.s = s
#         self.m = m

#         self.weight = Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)

#         self.easy_margin = easy_margin
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m

#     def forward(self, input, label):
#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         if self.device_id == None:
#             cosine = F.linear(F.normalize(input), F.normalize(self.weight))
#         else:
#             x = input
#             sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
#             temp_x = x.cuda(self.device_id[0])
#             weight = sub_weights[0].cuda(self.device_id[0])
#             cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
#             for i in range(1, len(self.device_id)):
#                 temp_x = x.cuda(self.device_id[i])
#                 weight = sub_weights[i].cuda(self.device_id[i])
#                 cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])),
#                                    dim=1)
#         cosine=cosine.float()
#         sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
#         phi = cosine * self.cos_m - sine * self.sin_m
#         phi=phi.float()
#         # pdb.set_trace()
#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#         else:
#             phi = torch.where(cosine > self.th, phi, cosine - self.mm)
#         # --------------------------- convert label to one-hot ---------------------------
#         # pdb.set_trace()
#         if len(label.shape)>1:
#             if self.device_id == None:
#                 one_hot=label.cuda()
#             else:
#                 one_hot=label.cuda(self.device_id[0])
#         else:
#             one_hot = torch.zeros(cosine.size())
#             if self.device_id != None:
#                 one_hot = one_hot.cuda(self.device_id[0])
#             else:
#                 in_device=label.device
#                 one_hot=one_hot.to(in_device)
#             one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         output = (one_hot * phi) + (
#                     (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
#         output *= self.s

#         return output


# class CosFace(nn.Module):
#     r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
#     Args:
#         in_features: size of each input sample
#         out_features: size of each output sample
#         device_id: the ID of GPU where the model will be trained by model parallel.
#                        if device_id=None, it will be trained on CPU without model parallel.
#         s: norm of input feature
#         m: margin
#         cos(theta)-m
#     """

#     def __init__(self, in_features, out_features, device_id, s=64.0, m=0.4):#0.35
#         super(CosFace, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.device_id = device_id
#         self.s = s
#         self.m = m
#         print("self.device_id", self.device_id)
#         self.weight = Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)

#     def forward(self, input, label):
#         # --------------------------- cos(theta) & phi(theta) ---------------------------

#         if self.device_id == None:
#             cosine = F.linear(F.normalize(input), F.normalize(self.weight))
#         else:
#             x = input
#             sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
#             temp_x = x.cuda(self.device_id[0])
#             weight = sub_weights[0].cuda(self.device_id[0])
#             cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
#             for i in range(1, len(self.device_id)):
#                 temp_x = x.cuda(self.device_id[i])
#                 weight = sub_weights[i].cuda(self.device_id[i])
#                 cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])),
#                                    dim=1)
#         phi = cosine - self.m
#         # --------------------------- convert label to one-hot ---------------------------
#         one_hot = torch.zeros(cosine.size())
#         # pdb.set_trace()
#         if len(label.shape)>1:
#             if self.device_id == None:
#                 one_hot=label.cuda()
#             else:
#                 one_hot=label.cuda(self.device_id[0])
#         else:
#             if self.device_id != None:
#                 one_hot = one_hot.cuda(self.device_id[0])
#             # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot

#                 one_hot.scatter_(1, label.cuda(self.device_id[0]).view(-1, 1).long(), 1)
#             else:
#                 in_device=label.device
#                 one_hot=one_hot.to(in_device)
#                 one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         output = (one_hot * phi) + (
#                     (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
#         output *= self.s

#         return output

#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#                + 'in_features = ' + str(self.in_features) \
#                + ', out_features = ' + str(self.out_features) \
#                + ', s = ' + str(self.s) \
#                + ', m = ' + str(self.m) + ')'

class ViT_face_landmark_patch8(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls',num_patches=None, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,fp16=True,with_land=False):
        super().__init__()#cls
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        if num_patches==None:
            num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # pdb.set_trace()
        self.with_land=with_land
        self.patch_size = patch_size
        self.fp16=fp16
        self.row_num=int(np.sqrt(num_patches)/2)#49
        self.row_num=int(np.sqrt(num_patches))#196
        if self.with_land:
            self.stn=MobileNetV3_backbone(mode='large')
            # # # pdb.set_trace()
            # # # self.stn= ViT_face_stn_patch8(
            # # #                  loss_type = 'None',
            # # #                  GPU_ID = GPU_ID,
            # # #                  num_class = num_class,
            # # #                  image_size=112,
            # # #                  patch_size=8,#8
            # # #                  dim=96,#512
            # # #                  depth=12,#20
            # # #                  heads=3,#8
            # # #                  mlp_dim=1024,
            # # #                  dropout=0.1,
            # # #                  emb_dropout=0.1
            # # #              )
            # # #resnet
            # # self.stn=models.resnet50()
            # # self.stn.fc=nn.Sequential()
            # # # hybrid_dimension=50
            # # # drop_ratio=0.9

            self.output_layer = nn.Sequential(
                nn.Dropout(p=0.5),    # refer to paper section 6
                nn.Linear(160, self.row_num*self.row_num*2),#2048
            )
            
            # self.output_layer = nn.Linear(96, int(self.row_num*self.row_num*2))#49*2,6        mobilenet 96 irse:128
            # self.patch_shape=torch.tensor([2*patch_size,2*patch_size])#49
            self.patch_shape=torch.tensor([patch_size,patch_size])#196
        self.theta=0

        # self.drop_2d=torch.nn.Dropout2d(p=0.1)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            # nn.Dropout(p=0.1),
            # nn.Linear(dim,512),
            nn.LayerNorm(dim),#nn.Identity()
        )
        self.sigmoid=nn.Sigmoid()
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID

        if self.loss_type == 'None':
            print("no loss for vit_face")
        else:
            if self.loss_type == 'Softmax':
                self.loss = Softmax(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'CosFace':

                self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID,m=0.4)
                # 
                # 
            elif self.loss_type == 'ArcFace':
                self.loss = ArcFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
                
            elif self.loss_type == 'AdaFace':
                self.loss=AdaFace(embedding_size=dim,classnum=num_class)

    def forward(self, x, label= None , mask = None,visualize=False,save_token=False,opt=None):
        p = self.patch_size
        

        if len(x.shape)==4:
            x=x/255.0*2-1  #no mean
            #test ms1m   #switch rgb
            # x=x[:,::-1,:,:]
            # x=x.flip(1)
            # x=torch.from_numpy(x.cpu().numpy()[:,::-1,:,:].copy()).cuda()
        # if self.with_land:
        if self.with_land:
            # x=x/255.0*2-1  #no mean
            theta=self.stn(x)#.forward(x)            #with original stn


            theta = theta.mean(dim=(-2, -1))#average pooling   for cnn
            theta=self.output_layer(theta)
            
            #min max scale
            t_max=torch.max(theta,1)[0]#.repeat(1,49*2)
            t_max=torch.unsqueeze(t_max,dim=1).repeat(1,self.row_num*self.row_num*2)
            t_min=torch.min(theta,1)[0]#.repeat(1,49*2)
            t_min=torch.unsqueeze(t_min,dim=1).repeat(1,self.row_num*self.row_num*2)
            theta=(theta-t_min)/(t_max-t_min)*111


            theta=theta.view(-1,self.row_num*self.row_num,2)
            self.theta=theta#.detach()

            x=extract_patches_pytorch_gridsample(x,theta,patch_shape=self.patch_shape,num_landm=int(self.row_num*self.row_num))
            
        if len(x.shape)==4:
            # x=x/255.0*2-1  #no mean

            x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        # else:
        #     x=X#.transpose(1,2)
        
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        # pdb.set_trace()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        # x=self.drop_2d(x)
        x = self.dropout(x)
        x = self.transformer(x, mask)
        if save_token==True:
            tokens=x[:,1:]
        x = x[:,1:,:].mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        emb = self.mlp_head(x)
        
        
        if save_token:
            return emb,tokens,self.theta
        if label is not None:
            if opt is not None:
                # pdb.set_trace()
                x = self.loss(emb, label,opt)
                return x, self.theta
            else:
                x = self.loss(emb, label)
                return x, self.theta
        # elif 

        else:
            if visualize==True:
                return emb,self.theta
            else:
                return emb
            # return emb


def extract_patches_pytorch_gridsample(imgs, landmarks, patch_shape,num_landm=49):#numpy
    """ Extracts patches from an image with gradient.
    Args:
        imgs: a numpy array of dimensions [batch_size, width, height, channels]
        landmarks: a numpy array of dimensions [batch_size, num_landm, 2]
        patch_shape: [width, height]
        num_landm: number of landmarks: int
    Returns:
        A reconstructed imaged with landmark only: [batch_size,channels, width, height]
    """
    # pdb.set_trace()
    device=landmarks.device

    img_shape=imgs.shape[2]
    # pdb.set_trace()
    list_patches = []
    patch_half_shape=patch_shape/2
    start = -patch_half_shape
    end = patch_half_shape
    
    sampling_grid = torch.meshgrid(torch.arange(start[0],end[0]),torch.arange(start[1],end[1]))
    sampling_grid=torch.stack(sampling_grid,dim=0).to(device)

    sampling_grid=torch.transpose(torch.transpose(sampling_grid,0,2),0,1)
    for i in range(num_landm):
        
        land=landmarks[:,i,:]

        patch_grid = (sampling_grid[None, :, :, :] + land[:, None, None, :])/(img_shape*0.5)-1
        sing_land_patch= F.grid_sample(imgs, patch_grid,align_corners=False)
        list_patches.append(sing_land_patch)
    # pdb.set_trace()
    list_patches=torch.stack(list_patches,dim=2)#.shape
    B, c, patches_num,w,h = list_patches.shape
    row=int(np.sqrt(patches_num))
    list_patches=list_patches.reshape(B,c,row,row,w,h)
    list_patches=list_patches.permute(0,1,2,4,3,5)
    list_patches=list_patches.reshape(B,c,w*int(np.sqrt(patches_num)),h*int(np.sqrt(patches_num)))
                      
    return list_patches

