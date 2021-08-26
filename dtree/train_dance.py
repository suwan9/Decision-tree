from __future__ import print_function
import yaml
import easydict
import os
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from apex import amp, optimizers
from data_loader.get_loader import get_loader
from utils.utils import *
from utils.lr_schedule import inv_lr_scheduler
from utils.loss import *
from models.LinearAverage import LinearAverage
from eval import test

from data_loader.get_loader import *
import datetime
from tensorboardX import SummaryWriter
from tqdm import tqdm
if is_in_notebook():
    from tqdm import tqdm_notebook as tqdm
#import torch.backends.cudnn as cudnn

# Training settings

import argparse


parser = argparse.ArgumentParser(description='Pytorch DA',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')

parser.add_argument('--source_path', type=str, default='./utils/source_list.txt', metavar='B',
                    help='path to source list')
parser.add_argument('--target_path', type=str, default='./utils/target_list.txt', metavar='B',
                    help='path to target list')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--exp_name', type=str, default='office_close', help='/path/to/config/file')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")

# args = parser.parse_args()
args = parser.parse_args()
config_file = args.config
conf = yaml.load(open(config_file))
save_config = yaml.load(open(config_file))
conf = easydict.EasyDict(conf)
#print(args.gpu_devices,'args.gpu_devices')
gpu_devices = ','.join([str(id) for id in args.gpu_devices])
#print(gpu_devices,'gpu_devices')
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

args.cuda = torch.cuda.is_available()
source_data = args.source_path
target_data = args.target_path
evaluation_data = args.target_path

batch_size = conf.data.dataloader.batch_size
filename = source_data.split("_")[1] + "2" + target_data.split("_")[1]
filename = os.path.join("record", args.exp_name,
                        config_file.replace(".yaml", ""), filename)
if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))
print("record in %s " % filename)


now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

log_dir = f'{conf.log.root_dir}/{now}'

logger = SummaryWriter(log_dir)



data_transforms = {
    source_data: transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    target_data: transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    evaluation_data: transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

use_gpu = torch.cuda.is_available()
source_loader, target_loader, \
test_loader, target_folder = get_loader(source_data, target_data,
                                        evaluation_data, data_transforms,
                                        batch_size=batch_size, return_id=True,
                                        balanced=conf.data.dataloader.class_balance)
dataset_test = test_loader
dataset_test1 = source_loader
n_share = conf.data.dataset.n_share
n_source_private = conf.data.dataset.n_source_private
num_class = n_share + n_source_private

G, C1 = get_model_mme(conf.model.base_model, num_class=num_class,
                      temp=conf.model.temp)
device = torch.device("cuda")

class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.feature_extractor = G
        #classifier_output_dim = len(source_classes)
        self.discriminator = AdversarialNetwork(2048)
        self.classifier = C1
        

    def forward(self, x):
        f = self.feature_extractor(x)
        d = self.discriminator(f)
        f, _, __, y = self.classifier(f)
        
        return y, d
totalNet = TotalNet()
#feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=gpu_ids, output_device=output_device).train(True)
#classifier1 = nn.DataParallel(totalNet.classifier, device_ids=gpu_ids, output_device=output_device).train(True)
#discriminator = nn.DataParallel(totalNet.discriminator, device_ids=gpu_ids, output_device=output_device).train(True)
if args.cuda:
    totalNet.feature_extractor.cuda()
    totalNet.classifier.cuda()
    totalNet.discriminator.cuda()
feature_extractor = totalNet.feature_extractor.to(device)
classifier = totalNet.classifier.to(device)
discriminator = totalNet.discriminator.to(device)
'''
if args.cuda:
    G.cuda()
    C1.cuda()
G.to(device)
C1.to(device)
'''
ndata = target_folder.__len__()

## Memory
lemniscate = LinearAverage(2048, ndata, conf.model.temp, conf.train.momentum).cuda()
params = []
for key, value in dict(feature_extractor.named_parameters()).items():
    if value.requires_grad and "features" in key:
        if 'bias' in key:
            params += [{'params': [value], 'lr': conf.train.multi,
                        'weight_decay': conf.train.weight_decay}]
        else:
            params += [{'params': [value], 'lr': conf.train.multi,
                        'weight_decay': conf.train.weight_decay}]
    else:
        if 'bias' in key:
            params += [{'params': [value], 'lr': 1.0,
                        'weight_decay': conf.train.weight_decay}]
        else:
            params += [{'params': [value], 'lr': 1.0,
                        'weight_decay': conf.train.weight_decay}]
criterion = torch.nn.CrossEntropyLoss().cuda()
criterion1 = torch.nn.BCELoss().cuda()

opt_g = optim.SGD(params, momentum=conf.train.sgd_momentum,
                  weight_decay=0.0005, nesterov=True)
'''
opt_c1 = optim.SGD(list(C1.parameters()), lr=1.0,
                   momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                   nesterov=True)
'''
fc_para = [{"params": C1.fc.parameters()}, {"params": C1.fc2.parameters()}]#,
           #{"params": C1.fc3.parameters()}, {"params": C1.fc4.parameters()},
           #{"params": C1.fc5.parameters()}]
opt_c1 = optim.SGD(fc_para, lr=1.0*2 , weight_decay=0.0005,
              momentum=conf.train.sgd_momentum, nesterov=True)#* 5
opt_discriminator = optim.SGD(discriminator.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9, nesterov=True)

'''
optimizer_fc = optim.SGD(fc_para, lr=args.train.lr * 5, weight_decay=args.train.weight_decay,
              momentum=args.train.momentum, nesterov=True)
'''
[feature_extractor, classifier, discriminator], [opt_g, opt_c1, opt_discriminator] = amp.initialize([feature_extractor, classifier, discriminator],
                                                                                                    [opt_g, opt_c1, opt_discriminator],
                                                                                                    opt_level="O1")
G = nn.DataParallel(feature_extractor)
C1 = nn.DataParallel(classifier)
D = nn.DataParallel(discriminator)
param_lr_g = []
for param_group in opt_g.param_groups:
    param_lr_g.append(param_group["lr"])
param_lr_f = []
for param_group in opt_c1.param_groups:
    param_lr_f.append(param_group["lr"])
param_lr_d = []
for param_group in opt_discriminator.param_groups:
    param_lr_d.append(param_group["lr"])


def train():
    best_acc = 0
    best_acc1 = 0
    best_acc2 = 0
    best_acc3 = 0
    best_acc4 = 0
    bscore3_all = 0
    bscore4_all = 0
    bscore1_all = 0
    bscore2_all = 0
    criterion = nn.CrossEntropyLoss().cuda()
    criterion1 = nn.BCELoss().cuda()
    print('train start!')
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)

    data_iter_s2 = iter(source_train_dl2)
    '''
    data_iter_s3 = iter(source_train_dl3)
    data_iter_s4 = iter(source_train_dl4)
    data_iter_s5 = iter(source_train_dl5)
    '''
    len_train_source2 = len(source_train_dl2)
    '''
    len_train_source3 = len(source_train_dl3)
    len_train_source4 = len(source_train_dl4)
    len_train_source5 = len(source_train_dl5)
    '''

    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    for step in range(conf.train.min_step + 1):
        G.train()
        C1.train()
        D.train()
        if step % 500 == 1 and step > 500:
            source_train_dst = FileListDataset(list_path='./CMUM1cy4.txt', path_prefix='',
                                  transform=data_transforms[source_data], filter=(lambda x: x in source_classes))

            source_train_dlt = DataLoader(dataset=source_train_dst, batch_size=36,
                             shuffle=True, num_workers=0, drop_last=True)
            data_iter_st = iter(source_train_dlt)
            len_train_sourcet = len(source_train_dlt)


        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)

        if step % len_train_source2 == 0:
            data_iter_s2 = iter(source_train_dl2)
        '''
        if step % len_train_source3 == 0:
            data_iter_s3 = iter(source_train_dl3)
        if step % len_train_source4 == 0:
            data_iter_s4 = iter(source_train_dl4)
        if step % len_train_source5 == 0:
            data_iter_s5 = iter(source_train_dl5)
        '''
        data_s2 = next(data_iter_s2)
        '''
        data_s3 = next(data_iter_s3)
        data_s4 = next(data_iter_s4)
        data_s5 = next(data_iter_s5)
        '''

        data_t = next(data_iter_t)
        data_s = next(data_iter_s)

        #data_st = next(data_iter_st)

        inv_lr_scheduler(param_lr_g, opt_g, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        inv_lr_scheduler(param_lr_f, opt_c1, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        inv_lr_scheduler(param_lr_d, opt_discriminator, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        img_s = data_s[0]
        label_s = data_s[1]
        img_t = data_t[0]
        index_t = data_t[2]

        img_s2 = data_s2[0]
        label_s2 = data_s2[1]
        '''
        img_s3 = data_s3[0]
        label_s3 = data_s3[1]
        img_s4 = data_s4[0]
        label_s4 = data_s4[1]
        img_s5 = data_s5[0]
        label_s5 = data_s5[1]
        '''
        img_s2, label_s2 = Variable(img_s2.cuda()), \
                         Variable(label_s2.cuda())
        '''
        img_s3, label_s3 = Variable(img_s3.cuda()), \
                         Variable(label_s3.cuda())
        img_s4, label_s4 = Variable(img_s4.cuda()), \
                         Variable(label_s4.cuda())
        img_s5, label_s5 = Variable(img_s5.cuda()), \
                         Variable(label_s5.cuda())
        '''

        img_s, label_s = Variable(img_s.cuda()), \
                         Variable(label_s.cuda())
        img_t = Variable(img_t.cuda())
        index_t = Variable(index_t.cuda())
        if len(img_t) < batch_size:
            break
        if len(img_s) < batch_size:
            break
        opt_g.zero_grad()
        opt_c1.zero_grad()
        opt_discriminator.zero_grad()
        ## Weight normalizztion
        C1.module.weight_norm()
        ## Source loss calculation

        if step > 500:
            if step % len_train_sourcet == 0:
                data_iter_st = iter(source_train_dlt)
            data_st = next(data_iter_st)
            img_st = data_st[0]
            label_st = data_st[1]
            img_st, label_st = Variable(img_st.cuda()), \
                                Variable(label_st.cuda())
            feat_st = G(img_st)
            feat_sts,out_st,_,_,_,_ = C1(feat_st)
            loss_st = criterion(out_st, label_st)
            

        feat = G(img_s)
        feats,out_s,_,_,_,_ = C1(feat)
        loss_s = criterion(out_s, label_s)

        feat2 = G(img_s2)
        '''
        feat3 = G(img_s3)
        feat4 = G(img_s4)
        feat5 = G(img_s5)
        '''
        _,_,out_s2,_,_,_ = C1(feat2)
        '''
        _,_,_,out_s3,_,_ = C1(feat3)
        _,_,_,_,out_s4,_ = C1(feat4)
        _,_,_,_,_,out_s5 = C1(feat5)
        '''
        loss_s2 = criterion(out_s2, label_s2)
        '''
        loss_s3 = criterion(out_s3, label_s3)
        loss_s4 = criterion(out_s4, label_s4)
        loss_s5 = criterion(out_s5, label_s5)
        '''


        feat_t = G(img_t)
        feat_tt,out_t,_,_,_,_ = C1(feat_t)
        feat_t = F.normalize(feat_t)
        ### Calculate mini-batch x memory similarity
        feat_mat = lemniscate(feat_t, index_t)
        ### We do not use memory features present in mini-batch
        feat_mat[:, index_t] = -1 / conf.model.temp
        ### Calculate mini-batch x mini-batch similarity
        feat_mat2 = torch.matmul(feat_t,
                                 feat_t.t()) / conf.model.temp
        mask = torch.eye(feat_mat2.size(0),
                         feat_mat2.size(0)).bool().cuda()
        feat_mat2.masked_fill_(mask, -1 / conf.model.temp)
        loss_nc = conf.train.eta * entropy(torch.cat([out_t, feat_mat,
                                                      feat_mat2], 1))
        loss_ent = conf.train.eta * entropy_margin(out_t, conf.train.thr,
                                                   conf.train.margin)
        
        adv_loss_separate = torch.zeros(1, 1).to(device)
        domain_prob_discriminator_source = D.forward(feat.cuda())
        domain_prob_discriminator_target = D.forward(feat_t.cuda())
        #domain_prob_discriminator_source = torch.from_numpy(domain_prob_discriminator_source).cuda()
        #domain_prob_discriminator_target = torch.from_numpy(domain_prob_discriminator_target).cuda()
        domain_prob_discriminator_source = domain_prob_discriminator_source.float().cuda()
        domain_prob_discriminator_target = domain_prob_discriminator_target.float().cuda()
        adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_source, torch.ones_like(domain_prob_discriminator_source))
        adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_target, torch.zeros_like(domain_prob_discriminator_target))
        
        #if step <= 5000:
        all = loss_nc + (loss_s+loss_s2)/2 + loss_ent + adv_loss_separate    #
        #+loss_s3+loss_s4+loss_s5
        #if step > 5000:
            #all = loss_nc + (loss_s+loss_s2)/2 + loss_ent + adv_loss_separate + loss_st
        
        #all = loss_nc + (loss_s+loss_s2)/2 + loss_ent + adv_loss_separate    #loss_nc + 
        with amp.scale_loss(all, [opt_g, opt_c1, opt_discriminator]) as scaled_loss:
            scaled_loss.backward()
        opt_g.step()
        opt_c1.step()
        opt_discriminator.step()
        opt_g.zero_grad()
        opt_c1.zero_grad()
        opt_discriminator.zero_grad()
        lemniscate.update_weight(feat_t, index_t)

        if step > 0 and step % conf.test.save_interval == 0 and step <= 2500:
            data = {
                "feature_extractor": G.state_dict(),
                'classifier': C1.state_dict(),
                'discriminator': D.state_dict() if not isinstance(D, Nonsense) else 1.0,
                #'discriminator_separate': discriminator_separate.state_dict(),
            }

            with open(os.path.join(log_dir, 'current.pkl'), 'wb') as f:
                torch.save(data, f)


        if step % conf.train.log_interval == 0:
            print('Train [{}/{} ({:.2f}%)]\tLoss Source: {:.6f} '
                  'Loss NC: {:.6f} Loss ENS: {:.6f}\t'.format(
                step, conf.train.min_step,
                100 * float(step / conf.train.min_step),
                loss_s.item(), loss_nc.item(), loss_ent.item()))
        if step > 0 and step % conf.test.test_interval == 0:
            
            ##############################
            '''
            totalnet = TotalNet()
            if args.cuda:
                totalnet.feature_extractor.cuda()
                totalnet.classifier.cuda()
                totalnet.discriminator.cuda()
            feature_extractor1 = totalnet.feature_extractor.to(device)
            classifier1 = totalnet.classifier.to(device)
            discriminator1 = totalnet.discriminator.to(device)
            
            params = []
            for key, value in dict(feature_extractor1.named_parameters()).items():
                if value.requires_grad and "features" in key:
                    if 'bias' in key:
                        params += [{'params': [value], 'lr': conf.train.multi,
                                    'weight_decay': conf.train.weight_decay}]
                    else:
                        params += [{'params': [value], 'lr': conf.train.multi,
                                    'weight_decay': conf.train.weight_decay}]
                else:
                    if 'bias' in key:
                        params += [{'params': [value], 'lr': 1.0,
                                    'weight_decay': conf.train.weight_decay}]
                    else:
                        params += [{'params': [value], 'lr': 1.0,
                                    'weight_decay': conf.train.weight_decay}]
            discriminator1 = nn.DataParallel(discriminator1)
            '''
            
            '''
            Gl, C1l = get_model_mme(conf.model.base_model, num_class=num_class,
                      temp=conf.model.temp)
            class TotalNet1(nn.Module):
                def __init__(self):
                    super(TotalNet1, self).__init__()
                    self.feature_extractor = Gl
                    #classifier_output_dim = len(source_classes)
                    
                    self.classifier = C1l
                    

                def forward(self, x):
                    f = self.feature_extractor(x)
                    
                    f, _, __, y = self.classifier(f)
                    
                    return y, d
            totalNet1 = TotalNet1()
            device1 = torch.device("cuda")

            if args.cuda:
                totalNet1.feature_extractor.cuda()
                totalNet1.classifier.cuda()
                
            Gl = totalNet1.feature_extractor.to(device)
            C1l = totalNet1.classifier.to(device)
            
            
            #ndata = target_folder.__len__()

            ## Memory
            #lemniscate = LinearAverage(2048, ndata, conf.model.temp, conf.train.momentum).cuda()
            
            paraml = []
            for key, value in dict(Gl.named_parameters()).items():
                if value.requires_grad and "features" in key:
                    if 'bias' in key:
                        paraml += [{'params': [value], 'lr': conf.train.multi,
                                    'weight_decay': conf.train.weight_decay}]
                    else:
                        paraml += [{'params': [value], 'lr': conf.train.multi,
                                    'weight_decay': conf.train.weight_decay}]
                else:
                    if 'bias' in key:
                        paraml += [{'params': [value], 'lr': 1.0,
                                    'weight_decay': conf.train.weight_decay}]
                    else:
                        paraml += [{'params': [value], 'lr': 1.0,
                                    'weight_decay': conf.train.weight_decay}]
            criterion = torch.nn.CrossEntropyLoss().cuda()
            
            opt_gl = optim.SGD(paraml, momentum=conf.train.sgd_momentum,
                            weight_decay=0.0005, nesterov=True)
            opt_c1l = optim.SGD(list(C1l.parameters()), lr=1.0,
                            momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                            nesterov=True)
            [Gl, C1l], [opt_gl, opt_c1l] = amp.initialize([Gl, C1l],
                                                    [opt_gl, opt_c1l],
                                                    opt_level="O1")
            Gl = nn.DataParallel(Gl)
            C1l = nn.DataParallel(C1l)
            param_lr_gl = []
            for param_group in opt_gl.param_groups:
                param_lr_gl.append(param_group["lr"])
            param_lr_fl = []
            for param_group in opt_c1l.param_groups:
                param_lr_fl.append(param_group["lr"])
            


            #counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
            loss_matrix = np.zeros((int(332), 10))##########超惨   564   332  1967
            step1 = 0
            
            for epoch in range(20):
                start_test = True
                for batch_idx, data in enumerate(dataset_test1):
                    #with torch.no_grad():
                    if 0 == 0:
                        Gl.train()
                        C1l.train()
                        #Gl.eval()
                        #C1l.eval()
                        #D.train()
                        step1 = step1 + 1
                        img_ts, label_ts, path_t = data[0], data[1], data[2]
                        #print(img_ts, label_ts,'img_ts, label_ts')
                        img_ts, label_ts = Variable(img_ts.cuda()), \
                                            Variable(label_ts.cuda())
                        #img_ts, label_ts = img_ts.cuda(), label_ts.cuda()
                        #for i, (im, label) in enumerate(tqdm(test_loader, desc='testing ')):
                            #img_ts, label_ts = Variable(im.cuda()), \
                                                #Variable(label.cuda())
                            #im = im.to(device1)
                            #label = label.to(device1)

                           
                        inv_lr_scheduler(param_lr_gl, opt_gl, step1,
                                        init_lr=conf.train.lr,
                                        max_iter=conf.train.min_step)
                        inv_lr_scheduler(param_lr_fl, opt_c1l, step1,
                                        init_lr=conf.train.lr,
                                        max_iter=conf.train.min_step)
                            
                        opt_gl.zero_grad()
                        opt_c1l.zero_grad()                        
                            ## Weight normalizztion
                        #C1l.module.weight_norm()
                           

                        feat_t = Gl(img_ts)
                        featts,out_ts,_,_,_,_ = C1l(feat_t)
                        #print(out_ts)
                        #feat_t = F.normalize(feat_t)
                        #out_tt = F.softmax(out_t)
                            #entr = -torch.sum(out_tt * torch.log(out_tt), 1).data.cpu().numpy()
                            #print(out_tt,'out_tt')
                            #pred = out_tt.data.max(1)[1]

                            #k = label_ts.data.size()[0]
                            #pred_cls = pred.cpu().numpy()
                            #pred = pred.cpu().numpy()
                        
                        loss_s = criterion(out_ts, label_ts)
                        
                        with amp.scale_loss(loss_s, [opt_gl, opt_c1l]) as scaled_loss:
                            scaled_loss.backward()
                        opt_gl.step()
                        opt_c1l.step()
                            
                        opt_gl.zero_grad()
                        opt_c1l.zero_grad()
                        
                for batch_idx, data in enumerate(dataset_test):
                    with torch.no_grad():
                    #if 0 == 0:
                        #Gl.train()
                        #C1l.train()
                        Gl.eval()
                        C1l.eval()
                        #D.train()
                        step1 = step1 + 1
                        img_ts, label_ts, path_t = data[0], data[1], data[2]
                        #print(img_ts, label_ts,'img_ts, label_ts')
                        #img_ts, label_ts = Variable(img_ts.cuda()), \
                                            #Variable(label_ts.cuda())
                        img_ts, label_ts = img_ts.cuda(), label_ts.cuda()
                        #for i, (im, label) in enumerate(tqdm(test_loader, desc='testing ')):
                            #img_ts, label_ts = Variable(im.cuda()), \
                                                #Variable(label.cuda())
                            #im = im.to(device1)
                            #label = label.to(device1)


                        feat_t = Gl(img_ts)
                        featts,out_ts,_,_,_,_ = C1l(feat_t)           

                        prob,_= torch.max(out_ts, dim=1)
                            #print(prob,'prob')

                                #target_share_weight = get_target_share_weight(domain_prob, fc2_s, domain_temperature=1.0,
                                                                            #class_temperature=1.0)
                        if(start_test):
                            all_values = prob.float().to(device1)
                                    #all_indices = indices.float().to(device1)
                            start_test = False
                        else:
                            all_values = torch.cat((all_values, prob.float().to(device1)), 0)
                                    #all_indices = torch.cat((all_indices, indices.float().to(device1)), 0)
                    
                all_values = torch.tensor(all_values).to(device1)
                loss = - torch.log(all_values)
                        #print(loss,'loss1')
                        #print(np.size(loss),'loss1')

                loss = loss.data.cpu().numpy()
                        #print(loss,'loss')
                        #print(np.size(loss),'loss')

                loss_matrix[:,epoch%10]=loss#(epoch%5)
                #print(loss_matrix,'loss_matrix')
            '''
            ##############################

            best_acc,best_acc1,best_acc2,bscore1_all,bscore2_all,best_acc3,best_acc4,bscore3_all,bscore4_all = test(step, dataset_test, filename, n_share, num_class, G, C1, D,
                 conf.train.thr,best_acc,best_acc1,best_acc2,bscore1_all,bscore2_all,conf,num_class,args,best_acc3,best_acc4,bscore3_all,bscore4_all)
            G.train()
            C1.train()
            D.train()
        
        

train()
