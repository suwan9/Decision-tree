from torchvision import models
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function, Variable
from easydl import *



class GradReverse(Function):
    @staticmethod
    def forward(ctx, x,lambd):
        ctx.save_for_backward(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambd=ctx.saved_tensors[0]
        return grad_output.neg()*lambd, None
def grad_reverse(x,lambd=1.0):
    return GradReverse.apply(x, Variable(torch.ones(1)*lambd).cuda())


class ResBase(nn.Module):
    def __init__(self, option='resnet50', pret=True, unit_size=100):
        super(ResBase, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        mod = list(model_ft.children())
        mod.pop()
        self.features = nn.Sequential(*mod)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.dim)
        return x


class ResClassifier_MME(nn.Module):
    def __init__(self, num_classes=12, input_size=2048, temp=0.05):
        super(ResClassifier_MME, self).__init__()

        self.bottleneck = nn.Linear(input_size, 256)
        self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.fc2 = nn.Linear(input_size, num_classes, bias=False)
        self.fc3 = nn.Linear(input_size, num_classes, bias=False)
        self.fc4 = nn.Linear(input_size, num_classes, bias=False)
        self.fc5 = nn.Linear(input_size, num_classes, bias=False)

        '''
        self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.fc2 = nn.Linear(input_size, num_classes, bias=False)
        self.fc3 = nn.Linear(input_size, num_classes, bias=False)
        self.fc4 = nn.Linear(input_size, num_classes, bias=False)
        self.fc5 = nn.Linear(input_size, num_classes, bias=False)
        '''
        self.tmp = temp

        #nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        #nn.init.xavier_normal_(self.fc3.weight)
        #nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
        #nn.init.kaiming_normal_(self.fc5.weight)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if return_feat:
            return x
        x = F.normalize(x)

        feature = self.bottleneck(x)
        x1 = self.fc(x)/self.tmp
        x2 = self.fc2(x)/self.tmp
        x3 = self.fc3(x)/self.tmp
        x4 = self.fc4(x)/self.tmp
        x5 = self.fc5(x)/self.tmp

        '''
        x1 = self.fc(x)/self.tmp
        x2 = self.fc2(x)/self.tmp
        x3 = self.fc3(x)/self.tmp
        x4 = self.fc4(x)/self.tmp
        x5 = self.fc5(x)/self.tmp
        '''

        return feature,x1,x2,x3,x4,x5

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))

    def weights_init(self, m):
        m.weight.data.normal_(0.0, 0.1)

class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork with a gredient reverse layer.
    its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` module.
    """
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        #self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        #x_ = self.grl(x)
        y = self.main(x)#(x_)
        return y