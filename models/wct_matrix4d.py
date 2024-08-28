import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentNetwork(nn.Module):
    def __init__(self,layer,matrixSize=32):
        super(ContentNetwork,self).__init__()
        self.linear = nn.Sequential(nn.Linear(32,64),  
                                    nn.ReLU(inplace=True),
                                    nn.Linear(64,matrixSize),
                                    )
        self.fc = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)
        self.matrixSize=matrixSize

    def forward(self,x):
        b,n,c=x.size() 
        x=x.reshape(-1,c).float()
        out = self.linear(x)
        out=out.reshape(b,n,self.matrixSize)
        out=out.permute(0,2,1)
        out = torch.bmm(out,out.transpose(1,2)).div(n)  
        out = out.view(b,-1)
        return self.fc(out)
    
class StyleNetwork(nn.Module):
    def __init__(self,layer,matrixSize=32):
        super(StyleNetwork,self).__init__()
        self.convs = nn.Sequential(nn.Conv2d(32,16,3,1,1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(16,matrixSize,3,1,1),
                                    )
        self.fc = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)

    def forward(self,x):
        out = self.convs(x)
        b,c,h,w = out.size()
        out = out.view(b,c,-1)
        out = torch.bmm(out,out.transpose(1,2)).div(h*w)
        out = out.view(out.size(0),-1)
        return self.fc(out)
        

class MulLayer(nn.Module):
    def __init__(self,layer,matrixSize=32):
        super(MulLayer,self).__init__()
        self.snet = StyleNetwork(layer, matrixSize)
        self.cnet = ContentNetwork(layer, matrixSize)
        self.matrixSize = matrixSize
        self.compress = nn.Conv2d(32,matrixSize,1,1,0)
        self.unzip = nn.Conv2d(matrixSize,32,1,1,0)
        self.transmatrix = None

    def forward(self,volume_cF,cF,sF,trans=True):
        
        volume_cF=volume_cF.unsqueeze(0)
        cb, cn,  cc = volume_cF.size() 
        cFF = volume_cF.permute(2, 0, 1) 
        cFF = cFF.unsqueeze(0).reshape(1, cc, -1)  
        cMean = torch.mean(cFF,dim=2,keepdim=True) 
        cMean = cMean.unsqueeze(3)  
        cF = cF - cMean  

        sb,sc,sh,sw = sF.size()
        sFF = sF.view(sb,sc,-1)  
        sMean = torch.mean(sFF,dim=2,keepdim=True) 
        sMean = sMean.unsqueeze(3) 
        sMeanC = sMean.expand_as(cF)  
        sMeanS = sMean.expand_as(sF)  
        sF = sF - sMeanS  
        compress_content = self.compress(cF)  
        b,c,h,w = compress_content.size()
        compress_content = compress_content.view(b,c,-1) 

        cMatrix = self.cnet(volume_cF) 
        sMatrix = self.snet(sF)  

        sMatrix = sMatrix.view(sMatrix.size(0), self.matrixSize, self.matrixSize) 
        cMatrix = cMatrix.view(cMatrix.size(0), self.matrixSize, self.matrixSize)  
        cMatrix = cMatrix.repeat(sMatrix.size(0), 1, 1)  
        transmatrix = torch.bmm(sMatrix, cMatrix) 
        transfeature = torch.bmm(transmatrix, compress_content).view(b, c, h, w)  
        out = self.unzip(transfeature.view(b, c, h, w))  
        out = out + sMeanC  
        return out, transmatrix

    def multi_style_interpolation(self,volume_cF,cF,sF_0,sF_1,sF_2,sF_3,weight,trans=True):
       
        volume_cF=volume_cF.unsqueeze(0)
        cb, cn,  cc = volume_cF.size()  
        cFF = volume_cF.permute(2, 0, 1) 
        cFF = cFF.unsqueeze(0).reshape(1, cc, -1)  
        cMean = torch.mean(cFF,dim=2,keepdim=True) 
        cMean = cMean.unsqueeze(3)  
        cF = cF - cMean  

        sb,sc,sh,sw = sF_0.size()
        sFF_0 = sF_0.view(sb,sc,-1) 
        sMean_0 = torch.mean(sFF_0,dim=2,keepdim=True)  
        sMean_0 = sMean_0.unsqueeze(3)  
        sMeanC_0 = sMean_0.expand_as(cF)  
        sMeanS_0 = sMean_0.expand_as(sF_0)  
        sF_0 = sF_0 - sMeanS_0 

        sb,sc,sh,sw = sF_1.size()
        sFF_1 = sF_1.view(sb,sc,-1) 
        sMean_1 = torch.mean(sFF_1,dim=2,keepdim=True)  
        sMean_1 = sMean_1.unsqueeze(3)  
        sMeanC_1 = sMean_1.expand_as(cF)  
        sMeanS_1 = sMean_1.expand_as(sF_1) 
        sF_1 = sF_1 - sMeanS_1  

        sb,sc,sh,sw = sF_2.size()
        sFF_2 = sF_2.view(sb,sc,-1)  
        sMean_2 = torch.mean(sFF_2,dim=2,keepdim=True)  
        sMean_2 = sMean_2.unsqueeze(3)  
        sMeanC_2 = sMean_2.expand_as(cF)  
        sMeanS_2 = sMean_2.expand_as(sF_2) 
        sF_2 = sF_2 - sMeanS_2  

        sb,sc,sh,sw = sF_3.size()
        sFF_3 = sF_3.view(sb,sc,-1) 
        sMean_3 = torch.mean(sFF_3,dim=2,keepdim=True) 
        sMean_3 = sMean_3.unsqueeze(3)  
        sMeanC_3 = sMean_3.expand_as(cF)  
        sMeanS_3 = sMean_3.expand_as(sF_3)  
        sF_3 = sF_3 - sMeanS_3  

        sMean_w = weight[0] * sMean_0 + weight[1] * sMean_1 + weight[2] * sMean_2 + weight[3] * sMean_3
        sMeanC_w = sMean_w.expand_as(cF) 

        compress_content = self.compress(cF)  
    
        b,c,h,w = compress_content.size()
        compress_content = compress_content.view(b,c,-1) 

        if(trans):
            cMatrix = self.cnet(volume_cF)  
            sMatrix_0 = self.snet(sF_0)  
            sMatrix_1 = self.snet(sF_1)  
            sMatrix_2 = self.snet(sF_2)  
            sMatrix_3 = self.snet(sF_3) 
            sMatrix_w = weight[0] * sMatrix_0 + weight[1] * sMatrix_1 + weight[2] * sMatrix_2 + weight[3] * sMatrix_3

            sMatrix_w = sMatrix_w.view(sMatrix_w.size(0), self.matrixSize, self.matrixSize)  
            cMatrix = cMatrix.view(cMatrix.size(0), self.matrixSize, self.matrixSize)  
            transmatrix = torch.bmm(sMatrix_w, cMatrix)  
            transfeature = torch.bmm(transmatrix, compress_content).view(b, c, h, w)  
            out = self.unzip(transfeature.view(b, c, h, w)) 
            out = out + sMeanC_w  
            return out, transmatrix
        else:
            out = self.unzip(compress_content.view(b,c,d,h,w))
            out = out + cMean
            return out
        
def calc_mean_std(x, eps=1e-8):
        """
        calculating channel-wise instance mean and standard variance
        x: shape of (N,C,*)
        """
        mean = torch.mean(x.flatten(2), dim=-1, keepdim=True) # size of (N, C, 1)
        std = torch.std(x.flatten(2), dim=-1, keepdim=True) + eps # size of (N, C, 1)
        
        return mean, std

class CNN(nn.Module):
    def __init__(self, matrixSize=32):
        super(CNN,self).__init__()
        self.convs = nn.Sequential(nn.Conv2d(32,64,3,1,1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64,64,3,1,1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64,matrixSize,3,1,1))
        self.fc = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)

    def forward(self,x):
        out = self.convs(x)
        b,c,h,w = out.size()
        out = out.view(b,c,-1)
        out = torch.bmm(out,out.transpose(1,2)).div(h*w)
        out = out.view(out.size(0),-1)
        return self.fc(out)
    

