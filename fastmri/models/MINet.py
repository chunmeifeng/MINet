from fastmri.models import common
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return SR_Branch(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y 

class LAM_Module(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.view(m_batchsize, -1, height, width)
        return out

class CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""
    def __init__(self, in_dim):
        super(CSAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        
        out = self.gamma*out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class SR_Branch(nn.Module):
    def __init__(self,n_resgroups,n_resblocks,n_feats,conv=common.default_conv):
        super(SR_Branch, self).__init__()
        
        self.n_resgroups =  n_resgroups   
        self.n_resblocks = n_resblocks   
        self.n_feats =  n_feats    
        kernel_size = 3
        reduction = 16  

        scale = 2
        rgb_range = 255
        n_colors = 1
        res_scale =0.1
        act = nn.ReLU(True)
        
        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats,n_feats, kernel_size)]#n_colors

        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.csa = CSAM_Module(n_feats)
        self.la = LAM_Module(n_feats)
        self.last_conv = nn.Conv2d(n_feats*(n_resgroups+1), n_feats, 3, 1, 1)
        self.last = nn.Conv2d(n_feats*2, n_feats, 3, 1, 1)
        self.last1 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.tail = nn.Sequential(*modules_tail)
        self.final = nn.Conv2d(n_feats, n_colors, 3, 1, 1)

    def forward(self, x):
        outputs = []
        x = self.head(x)
        outputs.append(x)

        res = x

        for name, midlayer in self.body._modules.items():
            res = midlayer(res)   

            if name=='0':
                res1 = res.unsqueeze(1)
            else:
                res1 = torch.cat([res.unsqueeze(1),res1],1)
            
            outputs.append(res1)

        out1 = res   
        res = self.la(res1)
        out2 = self.last_conv(res)    
        out1 = self.csa(out1)   
        out = torch.cat([out1, out2], 1)  
        res = self.last(out)  
        
        res += x

        outputs.append(res)

        x = self.tail(res)

        return outputs, x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
class Pred_Layer(nn.Module):
    def __init__(self, in_c=32):
        super(Pred_Layer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        x = self.enlayer(x)
        x = self.outlayer(x)
        return x

class MINet(nn.Module):
    def __init__(self, n_resgroups,n_resblocks, n_feats):
        super(MINet, self).__init__()

        self.n_resgroups = n_resgroups
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.net1 = SR_Branch(
            n_resgroups = self.n_resgroups,    
            n_resblocks = self.n_resblocks,   
            n_feats = self.n_feats,      
        )

        self.net2 = SR_Branch(
            n_resgroups = self.n_resgroups,   
            n_resblocks = self.n_resblocks,    
            n_feats = self.n_feats,       
        )

        main_net = SR_Branch(
            n_resgroups = self.n_resgroups,    
            n_resblocks = self.n_resblocks,    
            n_feats = self.n_feats,       
        )

	
        self.body = main_net.body
        self.csa = main_net.csa
        self.la = main_net.la
        self.last_conv = main_net.last_conv
        self.last = main_net.last
        self.last1 = main_net.last1
        self.tail = main_net.tail
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        nlayer = len(self.net1.body._modules.items())
        self.fusion_convs = nn.ModuleList([nn.Conv2d(128, 64, kernel_size=1, padding=0) for i in range(nlayer)])
        self.fusion_convsT1 = nn.ModuleList([nn.Conv2d(64, 32, kernel_size=1, padding=0) for i in range(nlayer)])
        self.fusion_convsT2 = nn.ModuleList([nn.Conv2d(64, 32, kernel_size=1, padding=0) for i in range(nlayer)])
        self.map_convs = nn.ModuleList([nn.Conv2d(64, 1, kernel_size=3, padding=1) for i in range(nlayer)])
        self.rgbd_global = Pred_Layer(32 * 2)

    def forward(self, x1, x2):

        x1 = self.net1.head(x1)
        x2 = self.net2.head(x2)

        x2 = self.tail(x2)

        resT1 = x1
        resT2 = x2

        t1s = []
        t2s = []

        for m1, m2,fusion_conv in zip(self.net1.body._modules.items(),self.net2.body._modules.items(), self.fusion_convs):   
            name1, midlayer1 = m1
            _, midlayer2 = m2

            resT1 = midlayer1(resT1)  
            resT2 = midlayer2(resT2)

            t1s.append(resT1.unsqueeze(1))
            t2s.append(resT2.unsqueeze(1))

            res = torch.cat([resT1,resT2],dim=1)
            res = fusion_conv(res)
            
            res_t1 = resT1
            res_t2_next = res+resT2
	    resT2 = res_t2_next

        out1T1 = res_t1   
        out1T2 = res_t2_next

        ts = t1s + t2s
        ts = torch.cat(ts,dim=1)
        res1_T2 = self.net1.la(ts)
        out2_T2 = self.net1.last_conv(res1_T2)

        out1T1 = self.net1.csa(out1T1)   
        out1T2 = self.net2.csa(out1T2)

        outT2 = torch.cat([out1T2, out2_T2], 1)  
        resT1 = self.net1.last1(out1T1)  
        resT2 = self.net2.last(outT2)
        
        resT1 += x1
        resT2 += x2

        x1 = self.net1.final(resT1)
        x2 = self.net2.final(resT2)
        return x1,x2#x1=pd  x2=pdfs
 
