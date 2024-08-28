import torch
import torch.nn as nn
import torch.nn.functional as F


class styleLoss(nn.Module):
    def forward(self,input,target):
        ib,ic,ih,iw = input.size()
        iF = input.view(ib,ic,-1)
        iMean = torch.mean(iF,dim=2)
        iCov = GramMatrix()(input)

        tb,tc,th,tw = target.size()
        tF = target.view(tb,tc,-1)
        tMean = torch.mean(tF,dim=2)
        tCov = GramMatrix()(target)

        loss = nn.MSELoss(size_average=False)(iMean,tMean) + nn.MSELoss(size_average=False)(iCov,tCov)
        return loss/tb


class GramMatrix(nn.Module):
    def forward(self,input):
        b, c, h, w = input.size()
        f = input.view(b,c,h*w) # bxcx(hxw)
        # torch.bmm(batch1, batch2, out=None)   #
        # batch1: bxmxp, batch2: bxpxn -> bxmxn #
        G = torch.bmm(f,f.transpose(1,2)) # f: bxcx(hxw), f.transpose: bx(hxw)xc -> bxcxc
        return G.div_(c*h*w)


class LossCriterion(nn.Module):
    def __init__(self,style_layers,content_layers,style_weight,content_weight):
        super(LossCriterion,self).__init__()

        self.style_layers = style_layers
        self.content_layers = content_layers
        self.style_weight = style_weight
        self.content_weight = content_weight

        self.styleLosses = [styleLoss()] * len(style_layers)
        self.contentLosses = [nn.MSELoss()] * len(content_layers)

    def forward(self,tF,sF,cF):
        # content loss
        totalContentLoss = 0
        for i,layer in enumerate(self.content_layers):
            cf_i = cF[layer]
            cf_i = cf_i.detach()
            tf_i = tF[layer]
            loss_i = self.contentLosses[i]
            totalContentLoss += loss_i(tf_i,cf_i)
        totalContentLoss = totalContentLoss * self.content_weight

        # style loss
        totalStyleLoss = 0
        for i,layer in enumerate(self.style_layers):
            sf_i = sF[layer]
            sf_i = sf_i.detach()
            tf_i = tF[layer]
            loss_i = self.styleLosses[i]
            totalStyleLoss += loss_i(tf_i,sf_i)
        totalStyleLoss = totalStyleLoss * self.style_weight
        loss = totalStyleLoss + totalContentLoss

        return loss,totalStyleLoss,totalContentLoss


### GRAD loss

class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x


class GradLoss(nn.Module):

    def __init__(self):
        super(GradLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return self.loss(output_grad, gt_grad), output_grad, gt_grad
    
    
class CorrelationLoss(nn.Module):
        def __init__(self):
            super(CorrelationLoss, self).__init__()
    
        def get_gray(self,x):
            
            ''' 
            Convert image to its gray one.
            '''
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            x_gray = x.mul(convert).sum(dim=1)
            return x_gray.unsqueeze(1)

        
        def calc_mean(self,img_pag,ssize):
            res = torch.zeros(ssize).cuda()
            W = 5
            for i in range(0,W):
                for j in range(0,W):
                    if (W-1-i)==0 and (W-1-j)==0:
                        res = res + img_pag[:,:,i:,j:]
                    elif (W-1-i)==0:
                        res = res + img_pag[:,:,i:,j:-(W-1-j)]
                    elif (W-1-j)==0:
                        res = res + img_pag[:,:,i:-(W-1-i),j:]
                    else:
                        res = res + img_pag[:,:,i:-(W-1-i),j:-(W-1-j)]

            return res/(W**2)
        
        
        def forward(self, pred, gt):
            pred_gray = self.get_gray(pred)
            gt_gray = self.get_gray(gt)
            
            pred_pad = F.pad(pred_gray, (2,2,2,2), "constant", 0)
            gt_pad = F.pad(gt_gray, (2,2,2,2), "constant", 0)
            
            ssize = gt_gray.size()
            
            EXY = self.calc_mean(pred_pad*gt_pad,ssize)
            EX = self.calc_mean(pred_pad,ssize)
            EY = self.calc_mean(gt_pad,ssize)
            sigma2X = self.calc_mean(pred_pad**2,ssize) - EX**2
            sigma2Y = self.calc_mean(gt_pad**2,ssize) - EY**2
            
            corr = torch.abs((EXY - EX*EY)/torch.sqrt(sigma2X*sigma2Y + 1e-4))
            corr_loss = (1 - corr)
            
            return torch.mean(corr_loss), corr
        
        
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        if w_x==1:
            count_h = self._tensor_size(x[:,:,1:,:])
            h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
            return 2*(h_tv/count_h)/batch_size

        if h_x==1:
            count_w = self._tensor_size(x[:,:,:,1:])
            w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
            return 2*(w_tv/count_w)/batch_size

        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


def image_bound_loss(image):
    cen_v = 0.5
    min_v = 0
    max_v = 1
    mask_min = image < min_v
    if mask_min.any():
        bottom_bound_loss = torch.mean((image[mask_min] - 0.5) ** 2)
    else:
        bottom_bound_loss = 0.
    mask_max = image > max_v
    if mask_max.any():
        up_bound_loss = torch.mean((image[mask_max] - 0.5) ** 2)
    else:
        up_bound_loss = 0.
    return up_bound_loss, bottom_bound_loss
    