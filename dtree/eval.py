
import torch
import torch.nn.functional as F
import numpy as np
import logging
from lib import *
from scipy.spatial.distance import cdist
from decision_treey import get_labels
from decision_treey import format_data
from sklearn.tree import DecisionTreeClassifier
from data_loader.get_loader import *

import yaml
import easydict
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
import sys



def test(step, dataset_test, filename, n_share, unk_class, G, C1, D, threshold,best_acc,
        best_acc1,best_acc2,bscore1_all,bscore2_all,conf,num_class,args,best_acc3,best_acc4,bscore3_all,bscore4_all):

    #print(unk_class,'unk_class')
    G.eval()
    C1.eval()
    D.eval()
    correct = 0
    correct_close = 0
    size = 0
    class_list = [i for i in range(n_share)]
    class_list.append(unk_class)
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    per_class_correct_cls = np.zeros((n_share + 1)).astype(np.float32)
    all_pred = []
    all_gt = []
    start_test = True
    all_confidece = list()
    all_consistency = list()
    all_entropy = list()
    path_all = list()
    pred1 = []
    acc_all = [i for i in range(9)]
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t, path_t = data[0], data[1], data[2]
            #print(path_t,'path_t')
            img_t, label_t = img_t.cuda(), label_t.cuda()
            feat = G(img_t)
            featt,out_t,out_s2,out_s3,out_s4,out_s5 = C1(feat)
            domain_prob = D.forward(feat)

            out_tt = F.softmax(out_t)
            out_tt1 = F.softmax(out_s2)

            entropy = get_entropy(out_tt,out_tt1,domain_temperature=1.0,
                                    class_temperature=1.0)#out_s3,out_s4,out_s5, 
            consistency = get_consistency(out_t,out_s2)#,out_s3,out_s4,out_s5
            confidence, indices = torch.max(out_tt, dim=1)
            #confidence2, indices2 = torch.max(out_tt1, dim=1)
            #print(indices,'indices')

            entr = -torch.sum(out_tt * torch.log(out_tt), 1).data.cpu().numpy()
            #print(np.mean(entr),'entr')
            pred = out_tt.data.max(1)[1]
            k = label_t.data.size()[0]
            pred_cls = pred.cpu().numpy()
            pred = pred.cpu().numpy()
            for i in range(len(pred)):
                pred1.append(pred[i]) 
            #print(pred1,'pred1')
            #pred1 = pred


            pred_unk = np.where(entr > threshold)
            pred[pred_unk[0]] = unk_class
            #print(pred,'pred')
            #print(np.size(pred),'np.size(pred)')
            #print(unk_class,'unk_class')
            all_gt += list(label_t.data.cpu().numpy())
            all_pred += list(pred)
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                correct_ind_close = np.where(pred_cls[t_ind[0]] == i)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_correct_cls[i] += float(len(correct_ind_close[0]))
                per_class_num[i] += float(len(t_ind[0]))
                correct += float(len(correct_ind[0]))
                correct_close += float(len(correct_ind_close[0]))
                #print(correct_ind,'correct_ind')
            size += k
            

            all_confidece.extend(confidence)
            all_consistency.extend(consistency)
            all_entropy.extend(entropy)
            path_all.extend(path_t)

            pred = torch.tensor(pred)
            entr = torch.from_numpy(entr)
            all_indices1 = torch.from_numpy(np.array(pred1))
            #print(all_indices1,'all_indices1')
            if(start_test):
                
                prob = pred.float().cuda()
                domain_p = domain_prob.float().cuda()
                all_values = confidence.float().cuda()
                all_indices = indices.cuda()
                #all_indices1 = pred2.cuda()
                all_f = feat.float().cuda()
                label_tt = label_t.cuda()
                all_sm = featt.float().cuda()
                all_sf= out_t.float().cuda()
                entr_all = entr.cuda()
                start_test = False
            else:
                prob = torch.cat((prob, pred.float().cuda()), 0)
                domain_p = torch.cat((domain_p, domain_prob.float().cuda()), 0)
                all_values = torch.cat((all_values, confidence.float().cuda()), 0)
                all_indices = torch.cat((all_indices, indices.cuda()), 0)
                #all_indices1 = torch.cat((all_indices1, pred2.cuda()), 0)
                all_f = torch.cat((all_f, feat.float().cuda()), 0)
                all_sm = torch.cat((all_sm, featt.float().cuda()), 0)
                all_sf = torch.cat((all_sf, out_t.float().cuda()), 0)
                label_tt = torch.cat((label_tt, label_t.cuda() ), 0)#  
                entr_all = torch.cat((entr_all, entr.cuda()), 0)#  

    
    #print(np.mean(entr_all.tolist()),'entr_all')
    per_class_acc = per_class_correct / per_class_num
    close_p = float(per_class_correct_cls.sum() / per_class_num.sum())
    #print('###############熵############################')
    
    #print(
        #'\nTest set including unknown classes:  Accuracy: {}/{} ({:.0f}%)  '
        #'({:.4f}%)\n'.format(
            #correct, size,
            #100. * correct / size, float(per_class_acc.mean())))
    
    output = [step, list(per_class_acc), 'per class mean acc %s'%float(per_class_acc.mean()),
              float(correct / size), 'closed acc %s'%float(close_p)]
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=filename, format="%(message)s")
    logger.setLevel(logging.INFO)
    #print(output)
    logger.info(output)

    if per_class_acc.mean() > best_acc1:
        best_acc1 = per_class_acc.mean()
    #print(best_acc1,'best_acc1')
    acc_all[0] = best_acc1

    all_confidece = normalize_weight(torch.tensor(all_confidece))
    all_consistency = normalize_weight(torch.tensor(all_consistency))
    # confidence = nega_weight(torch.tensor(confidence))
    all_entropy = normalize_weight(torch.tensor(all_entropy))
    #print(domain_p[0:157],'domain_p[0:157]')
    #print(np.mean(domain_p.tolist()[0:157]),np.mean(domain_p.tolist()[157:331]),np.mean(domain_p.tolist()),'domain_p')
    #print(np.mean(all_confidece.tolist()[0:157]),np.mean(all_confidece.tolist()[157:331]),np.mean(all_confidece.tolist()),'all_confidece')
    #print(np.mean(all_consistency.tolist()[0:157]),np.mean(all_consistency.tolist()[157:331]),np.mean(all_consistency.tolist()),'all_consistency')
    #print(np.mean(all_entropy.tolist()[0:157]),np.mean(all_entropy.tolist()[157:331]),np.mean(all_entropy.tolist()),'all_entropy')
    #print(np.mean(entr_all.tolist()[0:157]),np.mean(entr_all.tolist()[157:331]),np.mean(entr_all.tolist()),'entr_all')

    ######能量#############################
    def calculate_natigative_logit(list_val):
        total = 0
        T = 10
        for ele in range(0, len(list_val)):
            total = total + list_val[ele]
        return T * np.log(total)
    T = 10
    logit_t_energy = all_sf
    list_logit = [np.exp(x)/T for _,x in enumerate(logit_t_energy.cpu())]
            # -E(X)  值越大，表示其越是分布内的样本，否则表示其越是分布外的样本
    energy = [calculate_natigative_logit(x) for _,x in enumerate(list_logit)]
    energy = energy / np.log(80) 
    energy = torch.Tensor(energy)
    energy = energy.cuda()#f
    energy = normalize_weight(torch.tensor(energy))
    #print(np.size(energy.tolist()),'energy')
    #print(np.mean(energy.tolist()[0:157]),np.mean(energy.tolist()[157:331]),np.mean(energy.tolist()),'energy')


    ########聚类距离############################
    all_f1 = torch.tensor(all_sm).cuda()#.cuda()
    #print(all_sm.size(0),'feature.size(0)')
    #print(all_sm.size(1),'feature.size(1)')
    #print(__,'ff')
    #print(fc2_s,'fc2_s')
    #print(all_sm.size(0),'ff.size(0)')
    #print(all_sm.size(1),'ff.size(1)')
    #print(all_sf.size(0),'fc2_s.size(0)')
    #print(all_sf.size(1),'fc2_s.size(1)')        
    all_f1 = torch.cat((all_f1, torch.ones(all_f1.size(0), 1).cuda()), 1)
    all_f1 = (all_f1.t() / torch.norm(all_f1, p=2, dim=1)).t()

    all_f1 = all_f1.float().cuda().tolist()#numpy()
            #print(all_f1,'all_f1')
            #all_f1 = all_f1.cuda()

            #print(np.size(all_sm,0),np.size(all_sm,1),'all_sm')#498 10 all_sm
            #axis = 0，返回该二维矩阵的行数
    all_softmax = nn.Softmax(dim=1)(all_sf)  
            #print(np.size(all_softmax,0),np.size(all_softmax,1),'all_softmax')#498 10 all_softmax
    _, predict = torch.max(all_softmax, 1)
    K = all_sf.size(1)
            #分别进行总的特征计算
    aff = all_softmax.float().cpu().numpy()#.tolist()cpu().cuda()
            #print(aff,'aff')
            #print(np.size(aff,0),np.size(aff,1),'aff')#498 10 aff
    initc = aff.transpose().dot(all_f1)#transpose()函数的作用就是调换数组的行列值的索引值
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
            #print(initc,'initc')
            #print(np.size(initc,0),np.size(initc,1),'initc')#10 257 initc

            #h_dict = {}
            #h_dict_true = {}
            #thre_dis = {}
            #val_filter = {}
            #thre_filter_3= {}
            #pred_num1 = []
            #fft = {}
            #all_feat1 = [[random.random() for _ in range(2)]for _ in range(int(args.datasize))]
    #print(len(all_entropy),'len(all_entropy)')
    fft = [[random.random() for _ in range(7)]for _ in range(int(len(all_entropy)))]#args.datasize
    ftt = [[random.random() for _ in range(7)]for _ in range(int(len(all_entropy)))]
    ptt = [[random.random() for _ in range(2)]for _ in range(int(len(all_entropy)))]
    ltt = [[random.random() for _ in range(2)]for _ in range(int(len(all_entropy)))]
    lt = [[int(0) for _ in range(1)]for _ in range(int(len(all_entropy)))]
            #for cls in range(7):
                #fft = []
                #h_dict[cls] = [] 
                #h_dict_true[cls] = [] 
                #thre_dis[cls] = []
                #val_filter[cls] = []
                #thre_filter_3[cls] = []
    labelset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    dd2 = cdist(all_f1, initc[labelset], 'cosine')
    pred_label = dd2.argmin(axis=1)
    pred_num1 = [dd2.min(axis =1 ),dd2.argmin(axis = 1 )]
    pred_num1[0] = normalize_weight(torch.tensor(pred_num1[0]))

########################################
    #print(np.mean(pred_num1[0].tolist()[0:157]),np.mean(pred_num1[0].tolist()[157:331]),np.mean(pred_num1[0].tolist()),'pred_num1')
    allin = np.array(pred_num1[0].tolist()) - np.array(energy.tolist()) + np.array(all_entropy.tolist()) + np.array(all_consistency.tolist()) - np.array(all_confidece.tolist())   
    print(np.mean(allin.tolist()[0:157]),np.mean(allin.tolist()[157:331]),np.mean(allin.tolist()),'allin')  
    counters1 = [AccuracyCounter() for x in range(len(source_classes) + 1)]
    sort_index = sorted(allin)#将X中的元素从小到大排序后，提取对应的索引index
    pp = sort_index[int(conf.test.threshold * len(sort_index))]
    for i in range(len(allin)):
        if allin[i] >= pp:
            lt[i] = 1
        else:
            lt[i] = 0
    #print(lt,'lt')
    alll1 = 0
    allc1 = 0
    label_tt = label_tt.tolist()
    #print(all_indices1,'all_indices1')
    for (each_predict_prob, each_label, label_all) in zip(all_indices1, label_tt, lt):
        alll1 = alll1+1

        if each_label in source_classes:            
            counters1[each_label].Ntotal += 1.0

            if label_all == 0 and each_predict_prob == each_label:
                allc1= allc1+1
                counters1[each_label].Ncorrect += 1.0
        else:
            counters1[-1].Ntotal += 1.0
            #each_pred_id = np.argmax(each_predict_prob)
            if label_all == 1:
                allc1= allc1+1
                counters1[-1].Ncorrect += 1.0

    print('##############全判断标准############################')
    acc_tests1 = [x.reportAccuracy() for x in counters1 if not np.isnan(x.reportAccuracy())]
    print(acc_tests1,'test_all')
    acc_test1 = torch.ones(1, 1) * np.mean(acc_tests1)
    print(f'test_all accuracy_all is {acc_test1.item()}')

    correct1 = [x.Ncorrect for x in counters1]
    amount1 = [x.Ntotal for x in counters1]
    common_acc1 = np.sum(correct1[0:-1]) / np.sum(amount1[0:-1])
    outlier_acc1 = correct1[-1] / amount1[-1]

    print('common_acc1={}, outlier_acc1={}'.format(common_acc1, outlier_acc1))
    bscore1 = 2 / (1 / common_acc1 + 1 / outlier_acc1)
    print('hscore1={}'.format(bscore1))

    if acc_test1 > best_acc3:
        best_acc3 = acc_test1
        bscore3_all = bscore1
    print(best_acc3,'best_acc3')
    print(bscore3_all,'bscore3_all')
    acc_all[5] = best_acc3
    acc_all[6] = bscore3_all
       
    
    #print(allc1/alll1,'allin')
###################################################            
            #print(pred_num1[0],'pred_num1[0]')

    for i in range(len(all_entropy)):
        fft[i][0] = all_entropy[i].cuda().tolist()
        fft[i][1] = all_consistency[i].cuda().tolist()
        fft[i][2] = all_confidece[i].cuda().tolist()
        fft[i][3] = pred_num1[0][i].tolist()
        fft[i][4] = domain_p[i][0].tolist()
        fft[i][5] = energy[i].tolist()

        ftt[i][0] = all_entropy[i].cuda().tolist()
        ftt[i][1] = all_consistency[i].cuda().tolist()
        ftt[i][2] = all_confidece[i].cuda().tolist()
        ftt[i][3] = pred_num1[0][i].tolist()
        ftt[i][4] = domain_p[i][0].tolist()
        ftt[i][5] = energy[i].tolist()
        ftt[i][6] = lt[i]
        if prob[i] == unk_class:#mean(pred_num1[0]):
            fft[i][6] = 1.0 #all_label[i].tolist()
            ltt[i][1] = 1.0
            ltt[i][0] = 0.0
            #lt[i] = int(1)
        else:
            fft[i][6] = 0.0
            ltt[i][1] = 0.0
            ltt[i][0] = 1.0
            #lt[i] = int(0)
    #print(len(lt),'lt')
    with open("./CMUM1cy2.txt", "w") as output:
                #i = 0
        for i in range(int(len(all_entropy))):
            s = str(fft[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
            s = s.replace("'",'').replace(' ','') +'\n'  #去除单引号，逗号，每行末尾追加换行符  .replace(',','')
            output.write(s)
    labels1 = get_labels("./CMUM1cy2.txt")
                #print(labels1,'labels1')
    train_dataset,train_features1 = format_data("./CMUM1cy2.txt")
    clf = DecisionTreeClassifier().fit(train_dataset, labels1)
    

    train_dataset1,train_features1 = format_data("./CMUM1cy2.txt")
    aa = clf.predict(train_dataset1)
    #print(aa,'aa')
    counters1 = [AccuracyCounter() for x in range(len(source_classes) + 1)]
    '''
            for (each_indice, each_label, label_all) in zip(predict_prob, label, aa):
                if each_label in source_classes:
                    counters1.add_total(each_label)
                    if label_all == '0.0' and each_indice == each_label:
                        counters1.add_correct(each_label)
                else:
                    counters1.add_total(-1)
                    if label_all == '1.0':
                        counters1.add_correct(-1)
      
    
    print(label_tt,'label_tt')
    
    for i in range(len(label_tt)):
        label_tt[i] = torch.from_numpy(np.array(label_tt[i]))   
    print(label_tt,'label_tt')   
    '''
    alll1 = 0
    allc1 = 0
    #label_tt = label_tt.tolist()
    for (each_predict_prob, each_label, label_all) in zip(all_indices1, label_tt, aa):
        alll1 = alll1+1

        if each_label in source_classes:
            #print(each_label,'each_label')
            counters1[each_label].Ntotal += 1.0
            #each_pred_id = np.argmax(each_predict_prob)

                    #if not outlier(each_target_share_weight, each_pred_id):
                        #counters[int(each_pred_id)].Npred += 1.0

            if label_all == '0.0' and each_predict_prob == each_label:
                allc1= allc1+1
                counters1[each_label].Ncorrect += 1.0
        else:
            counters1[-1].Ntotal += 1.0
            #each_pred_id = np.argmax(each_predict_prob)
            if label_all == '1.0':
                allc1= allc1+1
                counters1[-1].Ncorrect += 1.0
    '''
    print('##############熵决策树############################')
    acc_tests1 = [x.reportAccuracy() for x in counters1 if not np.isnan(x.reportAccuracy())]
    print(acc_tests1,'acc_tests1')
    acc_test1 = torch.ones(1, 1) * np.mean(acc_tests1)

    print(f'test1 accuracy1 is {acc_test1.item()}')
    #print(allc1/alll1,'allc1/alll1')

    correct1 = [x.Ncorrect for x in counters1]
    amount1 = [x.Ntotal for x in counters1]
    common_acc1 = np.sum(correct1[0:-1]) / np.sum(amount1[0:-1])
    outlier_acc1 = correct1[-1] / amount1[-1]

    print('common_acc1={}, outlier_acc1={}'.format(common_acc1, outlier_acc1))
    bscore1 = 2 / (1 / common_acc1 + 1 / outlier_acc1)
    print('hscore1={}'.format(bscore1))
    '''
    if acc_test1 > best_acc:
        best_acc = acc_test1
        bscore1_all = bscore1
    #print(best_acc,'best_acc')
    #print(bscore1_all,'bscore1_all')
    acc_all[1] = best_acc
    acc_all[3] = bscore1_all

########################
    '''
    Gl, C1l = get_model_mme(conf.model.base_model, num_class=2,
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
    device = torch.device("cuda")

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
    loss_matrix = np.zeros((int(len(all_entropy)), 10))##########超惨   564   332  1967
    step1 = 0
            
    for epoch in range(20):
        start_test = True
        sp = 0 
        ll = 0
        lop = 0
        for batch_idx, data in enumerate(dataset_test):
                    #with torch.no_grad():
            if 0 == 0:
                Gl.train()
                C1l.train()
                step1 = step1 + 1
                ll = sp*16
                sp = sp + 1 
                img_ts, label_ts, path_t = data[0], data[1], data[2]
                for i in range(len(label_ts.tolist())):
                    #a = lt[ll+i]
                    label_ts[i] = lt[ll+i]
                
                img_ts, label_ts = Variable(img_ts.cuda()), \
                                    Variable(label_ts.cuda())

                inv_lr_scheduler(param_lr_gl, opt_gl, step1,
                                init_lr=conf.train.lr,
                                max_iter=conf.train.min_step)
                inv_lr_scheduler(param_lr_fl, opt_c1l, step1,
                                init_lr=conf.train.lr,
                                max_iter=conf.train.min_step)
                            
                opt_gl.zero_grad()
                opt_c1l.zero_grad()                        
                feat_t = Gl(img_ts)
                featts,out_ts,_,_,_,_ = C1l(feat_t)
                
                        
                loss_s = criterion(out_ts, label_ts)
                lop = lop + loss_s
                
                        
                with amp.scale_loss(loss_s, [opt_gl, opt_c1l]) as scaled_loss:
                    scaled_loss.backward()
                opt_gl.step()
                opt_c1l.step()
                            
                opt_gl.zero_grad()
                opt_c1l.zero_grad()
        sp1 = 0 
        ll1 = 0  
        print(lop,'lop')               
        for batch_idx, data in enumerate(dataset_test):
            with torch.no_grad():
                    #if 0 == 0:
                        #Gl.train()
                        #C1l.train()
                Gl.eval()
                C1l.eval()
                        #D.train()
                #step1 = step1 + 1
                ll1 = sp1*16
                sp1 = sp1 + 1
                img_ts, label_ts, path_t = data[0], data[1], data[2]
                        #print(img_ts, label_ts,'img_ts, label_ts')
                        #img_ts, label_ts = Variable(img_ts.cuda()), \
                                            #Variable(label_ts.cuda())
                
                lt1 = [[int(0) for _ in range(1)]for _ in range(int(len(label_ts.tolist())))]
                for i in range(len(label_ts.tolist())):
                    lt1[i] = int(lt[ll1+i])
                    label_ts[i] = lt[ll1+i]
                lt1 = torch.IntTensor(lt1)
                
                img_ts, label_ts = img_ts.cuda(), label_ts.cuda()
                        #for i, (im, label) in enumerate(tqdm(test_loader, desc='testing ')):
                            #img_ts, label_ts = Variable(im.cuda()), \
                                                #Variable(label.cuda())
                            #im = im.to(device1)
                            #label = label.to(device1)


                feat_t = Gl(img_ts)
                featts,out_ts,_,_,_,_ = C1l(feat_t) 
                out_tt = F.softmax(out_ts)          

                prob,_= torch.max(out_ts, dim=1)
                            #print(prob,'prob')

                                #target_share_weight = get_target_share_weight(domain_prob, fc2_s, domain_temperature=1.0,
                                                                            #class_temperature=1.0)
                if(start_test):
                    all_values = out_tt.float().to(device)
                    all_value = prob.float().to(device)
                                    #all_indices = indices.float().to(device1)
                    start_test = False
                else:
                    all_values = torch.cat((all_values, out_tt.float().to(device)), 0)
                    all_value = torch.cat((all_value, prob.float().to(device)), 0)
                                    #all_indices = torch.cat((all_indices, indices.float().to(device1)), 0)
                    
        all_values = torch.tensor(all_values).to(device)
        all_value = torch.tensor(all_value).to(device)
        loss = - torch.log(all_value)
        #print(all_values,'all_values')
        for i in range(len(loss)):
            #print(all_values[i][0],all_values[i][1],'all_values')
            #print(ltt[i][0],ltt[i][1],'ltt')
            loss[i] = -(ltt[i][0]*torch.log(all_values[i][0]) + ltt[i][1]*torch.log(all_values[i][1]))

        #print(loss,'loss')
        
                        #print(loss,'loss1')
                        #print(np.size(loss),'loss1')

        loss = loss.data.cpu().numpy()
        #print(loss,'loss')
                        #print(np.size(loss),'loss')

        loss_matrix[:,epoch%10]=loss#(epoch%5)
                #print(loss_matrix,'loss_matrix')
    '''      
##############################


##############################
    '''        
    Gl, C1l = get_model_mme(conf.model.base_model, num_class=num_class,
              temp=conf.model.temp)
    device = torch.device("cuda")
    class TotalNet(nn.Module):
        def __init__(self):
            super(TotalNet, self).__init__()
            self.feature_extractor = Gl
                    #classifier_output_dim = len(source_classes)
                    
            self.classifier = C1l
                    

        def forward(self, x):
            f = self.feature_extractor(x)
                    
            f, _, __, y = self.classifier(f)
                    
            return y, d
    totalNet = TotalNet()
    #device1 = torch.device("cuda")

    if args.cuda:
        totalNet.feature_extractor.cuda()
        totalNet.classifier.cuda()
                
    Gl = totalNet.feature_extractor.to(device)
    C1l = totalNet.classifier.to(device)
            

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
        sp = 0 
        ll = 0 
        for batch_idx, data in enumerate(dataset_test):
                    #with torch.no_grad():
            if 0 == 0:
                Gl.train()
                C1l.train()
                        #Gl.eval()
                        #C1l.eval()
                        #D.train()
                ll = ll + sp*32
                sp = sp + 1
                step1 = step1 + 1
                img_ts, label_ts, path_t = data[0], data[1], data[2]
                print(len(label_ts.tolist()),'img_ts, label_ts')
                #lt = torch.Tensor(lt)
                #print(ltt,'ltt')
                print(lt,'lt')
                lt1 = [[random.random() for _ in range(1)]for _ in range(int(len(label_ts.tolist())))]
                for i in range(len(label_ts.tolist())):
                    lt1[i] = int(lt[ll+i])
                lt1 = torch.Tensor(lt1)
                print(lt1,'lt1')
                img_ts, label_ts = Variable(img_ts.cuda()), \
                                    Variable(lt1.cuda())#label_ts
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
                print(out_ts,'out_ts')
                            #pred = out_tt.data.max(1)[1]

                            #k = label_ts.data.size()[0]
                            #pred_cls = pred.cpu().numpy()
                            #pred = pred.cpu().numpy()
                out_ts = int(out_ts)  
                print(out_ts,'out_ts')      
                loss_s = criterion(out_ts, label_ts)
                        
                with amp.scale_loss(loss_s, [opt_gl, opt_c1l]) as scaled_loss:
                    scaled_loss.backward()
                opt_gl.step()
                opt_c1l.step()
                            
                opt_gl.zero_grad()
                opt_c1l.zero_grad()
        sp1 = 0 
        ll1 = 0                 
        for batch_idx, data in enumerate(dataset_test):
            with torch.no_grad():
                    #if 0 == 0:
                        #Gl.train()
                        #C1l.train()
                Gl.eval()
                C1l.eval()
                        #D.train()
                ll1 = ll1 + sp1*32
                sp1 = sp1 + 1
                img_ts, label_ts, path_t = data[0], data[1], data[2]
                        #print(img_ts, label_ts,'img_ts, label_ts')
                        #img_ts, label_ts = Variable(img_ts.cuda()), \
                                            #Variable(label_ts.cuda())
                lt1 = [[random.random() for _ in range(1)]for _ in range(int(len(label_ts.tolist())))]
                for i in range(len(label_ts.tolist())):
                    lt1[i] = int(lt[ll+i])
                lt1 = torch.Tensor(lt1)
                img_ts, label_ts = img_ts.cuda(), lt1.cuda()
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
        print(all_values,'all_values') 
                   
        all_values = torch.tensor(all_values).to(device1)
        loss = - torch.log(all_values)

        print(loss,'loss1')
        for i in range(len(loss)):
            loss[i] = criterion(all_values[i],ltt[i])
                        #print(np.size(loss),'loss1')

        loss = loss.data.cpu().numpy()
        print(loss,'loss')
                        #print(np.size(loss),'loss')

        loss_matrix[:,epoch%10]=loss#(epoch%5)
                #print(loss_matrix,'loss_matrix')
    '''        
##############################


########################
    '''
    loss_sele = loss_matrix[:, :10]
    #print(loss_matrix,'loss_matrix')
    #print(np.size(loss_matrix,0),np.size(loss_matrix,1),'loss_matrix')
    #print(loss_sele,'loss_sele')
    #print(np.size(loss_sele[0]),'loss_sele')

    loss_mean = loss_sele.mean(axis=1)
    #print(loss_mean,'loss_mean')
    #print(np.size(loss_mean),'np.size(loss_mean)')

    
    #cr0 = 0.25########################超惨
    #cr1 = 0.8
    sort_index = np.argsort(loss_mean)#将X中的元素从小到大排序后，提取对应的索引index
            #print(cr,'cr')
    #print(sort_index,'sort_index')
            #sort samples per class
    all1 = 0
    accc = 0
    for i in sort_index:
        all1 = all1 + 1
        if label_tt[i] > 9:
            if entr_all[i] > threshold:#if all_score[i] < args.threshold:#[i] == '1.0':
                accc = accc + 1
        else:
            if entr_all[i] <= threshold:#if all_score[i] >= args.threshold:#aa[i] == '0.0':
                accc = accc + 1
    print(accc/all1,'accc/all全-伪标记01准确率')

    clean_index = []
    clean_index1 = []
    cr = 0.5
    
           
    #for i in range(int(31)):########################超惨
    c = []
    b = i
    for idx in sort_index:
        if fft[idx][6] == 0.0:
            c.append(idx)
    clean_num = int(len(c)*cr)
    clean_idx = c[:clean_num]
    
    
        #for i in range(len(clean_idx)):
            #clean_index1.append(clean_idx[i]) 
        
        #print(b,'###################')
        #for i in clean_idx:            
            #print(label_tt[i],fft[i][6],'clean_idx')
    
    
    clean_index.extend(clean_idx)
        
    #print(np.size(label),'label')
    clean_idx = []
    cc = 0
    for idx in sort_index:
        if fft[idx][6] == 1.0:
            cc = cc+1
            clean_idx.append(idx)
    clean_index.extend(clean_idx[int(len(clean_idx)*0.5):len(clean_idx)])
    
    
    print('clean_index###################')
    for i in clean_index:            
        print(label_tt[i],fft[i][6],'clean_index')
    

    
    c = []
    cr0 = 0.4

    #for i in range(int(31)):########################超惨
    c = []
        #b = i    
    for idx in sort_index:
        if fft[idx][6] == 0.0:
            c.append(idx)
    clean_num = int(len(c)*cr0)
    clean_idx1 = c[:clean_num]
    clean_index1.extend(clean_idx1)
    
    
    print(np.size(clean_index),np.size(clean_index1),'np.size(clean_index)')
    
    #for i in clean_index1:            
        #print(label_tt[i],all_indices1[i].tolist(),'clean_index1')
    #for i in clean_idx:    
        #fft[i][6] = 0.0
    #clean_index.extend(clean_idx)

    #clean_num = int(len(c)*cr1)
    #clean_idx = c[clean_num:len(c)]
    #for i in clean_idx:    
        #fft[i][6] = 1.0
    #clean_index.extend(clean_idx)

    #for i in clean_index:    
        #print(label[i],fft[i][6],'clean_idx')  
            
    dd = 0
    for i in clean_index1:
        ptt[i][0] = path_all[i]
        ptt[i][1] = all_indices1[i].tolist()

    for i in clean_index:
        ftt[i] = fft[i]
                
        dd = dd + 1
            #print(ftt,'ftt')
    all1 = 0
    accc = 0
    for i in clean_index:
        all1 = all1 + 1
        if label_tt[i] > 9:
            if entr_all[i] > threshold:#if all_score[i] < args.threshold:#[i] == '1.0':
                accc = accc + 1
        else:
            if entr_all[i] <= threshold:#if all_score[i] >= args.threshold:#aa[i] == '0.0':
                accc = accc + 1
    print(accc/all1,'accc/all1-clean伪标记01准确率')
    '''
#############################



#################################
    sort_index = np.argsort(allin)#将X中的元素从小到大排序后，提取对应的索引index

    all1 = 0
    accc = 0
    for i in sort_index:
        all1 = all1 + 1
        if label_tt[i] > 9:
            if allin[i] > (2*np.mean(allin.tolist())): #if all_score[i] < args.threshold:#[i] == '1.0':
                accc = accc + 1
        else:
            if allin[i] <= (2*np.mean(allin.tolist())):#if all_score[i] >= args.threshold:#aa[i] == '0.0':
                accc = accc + 1
    print(accc/all1,'accc/all全-伪标记01准确率')

    clean_index = []
    clean_index1 = []
    c = []
    for idx in sort_index:
        if ftt[idx][6] == 0:
            c.append(idx)
    clean_num = int(len(c)*0.5)
    clean_idx = c[:clean_num]
    
    clean_index.extend(clean_idx)
        
    print(np.size(clean_index),'clean_index')
    clean_idx = []
    cc = 0
    for idx in sort_index:
        if ftt[idx][6] == 1:
            #cc = cc+1
            clean_idx.append(idx)
    clean_index.extend(clean_idx[int(len(clean_idx)*0.45):len(clean_idx)])
    #for i in clean_index:            
        #print(label_tt[i],fft[i][6],'clean_idx')
    print(np.size(clean_index),'clean_index')
    '''
    c = []
    cr0 = 0.4
    for idx in sort_index:
        if fft[idx][6] == 0.0:
            c.append(idx)
    clean_num = int(len(c)*cr0)
    clean_idx1 = c[:clean_num]
    clean_index1.extend(clean_idx1)
    for i in clean_index1:
        ptt[i][0] = path_all[i]
        ptt[i][1] = all_indices1[i].tolist()
    '''
    all1 = 0
    accc = 0
    for i in clean_index:
        all1 = all1 + 1
        if label_tt[i] > 9:
            if allin[i] > (2*np.mean(allin.tolist())):#if all_score[i] < args.threshold:#[i] == '1.0':
                accc = accc + 1
        else:
            if allin[i] <= (2*np.mean(allin.tolist())):#if all_score[i] >= args.threshold:#aa[i] == '0.0':
                accc = accc + 1
    print(accc/all1,'accc/all1-clean伪标记01准确率')

    with open("./CMUM1cy3y.txt", "w") as output:
                #i = 0
        for i in clean_index:
            s = str(ftt[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
            s = s.replace("'",'').replace(' ','') +'\n'  #去除单引号，逗号，每行末尾追加换行符  .replace(',','')
            output.write(s)


    labels1 = get_labels("./CMUM1cy3y.txt")
                #print(labels1,'labels1')
    train_dataset,train_features1 = format_data("./CMUM1cy3y.txt")
    clf = DecisionTreeClassifier().fit(train_dataset, labels1)
    

    train_dataset1,train_features1 = format_data("./CMUM1cy2.txt")
    aa1 = clf.predict(train_dataset1)
    #print(aa1,'aa1')
    counters2 = [AccuracyCounter() for x in range(len(source_classes) + 1)]
    
    #label_tt = label_tt.tolist()
    alll2 = 0
    allc2 = 0
    for (each_predict_prob, each_label, label_all) in zip(all_indices1, label_tt, aa1):
        alll2 = alll2+1
        if each_label in source_classes:
            #print(each_label,'each_label')
            counters2[each_label].Ntotal += 1.0
            #each_pred_id = np.argmax(each_predict_prob)

                    #if not outlier(each_target_share_weight, each_pred_id):
                        #counters[int(each_pred_id)].Npred += 1.0

            if label_all == '0' and each_predict_prob == each_label:
                allc2= allc2+1
                counters2[each_label].Ncorrect += 1.0
        else:
            counters2[-1].Ntotal += 1.0
            #each_pred_id = np.argmax(each_predict_prob)
            if label_all == '1':
                allc2 = allc2+1
                counters2[-1].Ncorrect += 1.0

    print('##############全判别标准——可信样本决策树############################')
    acc_tests2 = [x.reportAccuracy() for x in counters2 if not np.isnan(x.reportAccuracy())]
    print(acc_tests2,'acc_tests2')
    acc_test2 = torch.ones(1, 1) * np.mean(acc_tests2)

    print(f'test1 accuracy1 is {acc_test2.item()}')
    #print(allc2/alll2,'allc2/alll2')

    correct2 = [x.Ncorrect for x in counters2]
    amount2 = [x.Ntotal for x in counters2]
    common_acc2 = np.sum(correct2[0:-1]) / np.sum(amount2[0:-1])
    outlier_acc2 = correct2[-1] / amount2[-1]

    print('common_acc1={}, outlier_acc1={}'.format(common_acc2, outlier_acc2))
    bscore2 = 2 / (1 / common_acc2 + 1 / outlier_acc2)

    print('hscore1={}'.format(bscore2))
    if acc_test2 > best_acc4:
        best_acc4 = acc_test2
        bscore4_all = bscore2
    print(best_acc4,'best_acc4')
    print(bscore4_all,'bscore4_all')
    acc_all[7] = best_acc4
    acc_all[8] = bscore4_all
    

##########################



#################################
    sort_index = np.argsort(entr_all.cpu())#将X中的元素从小到大排序后，提取对应的索引index

    all1 = 0
    accc = 0
    for i in sort_index:
        all1 = all1 + 1
        if label_tt[i] > 9:
            if entr_all[i] > threshold:#if all_score[i] < args.threshold:#[i] == '1.0':
                accc = accc + 1
        else:
            if entr_all[i] <= threshold:#if all_score[i] >= args.threshold:#aa[i] == '0.0':
                accc = accc + 1
    print(accc/all1,'accc/all全-伪标记01准确率')

    clean_index = []
    clean_index1 = []
    c = []
    for idx in sort_index:
        if fft[idx][6] == 0.0:
            c.append(idx)
    clean_num = int(len(c)*0.50)
    clean_idx = c[:clean_num]
    
    clean_index.extend(clean_idx)
        
    print(np.size(clean_index),'clean_index')
    clean_idx = []
    cc = 0
    for idx in sort_index:
        if fft[idx][6] == 1.0:
            #cc = cc+1
            clean_idx.append(idx)
    clean_index.extend(clean_idx[int(len(clean_idx)*0.45):len(clean_idx)])
    #for i in clean_index:            
        #print(label_tt[i],fft[i][6],'clean_idx')
    print(np.size(clean_index),'clean_index')

    c = []
    cr0 = 0.4
    for idx in sort_index:
        if fft[idx][6] == 0.0:
            c.append(idx)
    clean_num = int(len(c)*cr0)
    clean_idx1 = c[:clean_num]
    clean_index1.extend(clean_idx1)
    for i in clean_index1:
        ptt[i][0] = path_all[i]
        ptt[i][1] = all_indices1[i].tolist()

    all1 = 0
    accc = 0
    for i in clean_index:
        all1 = all1 + 1
        if label_tt[i] > 9:
            if entr_all[i] > threshold:#if all_score[i] < args.threshold:#[i] == '1.0':
                accc = accc + 1
        else:
            if entr_all[i] <= threshold:#if all_score[i] >= args.threshold:#aa[i] == '0.0':
                accc = accc + 1
    print(accc/all1,'accc/all1-clean伪标记01准确率')


#################################

    with open("./CMUM1cy3.txt", "w") as output:
                #i = 0
        for i in clean_index:
            s = str(fft[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
            s = s.replace("'",'').replace(' ','') +'\n'  #去除单引号，逗号，每行末尾追加换行符  .replace(',','')
            output.write(s)
    with open("./CMUM1cy4.txt", "w") as output:
                #i = 0
        for i in clean_index1:
            s = str(ptt[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
            s = s.replace("'",'').replace(',','') +'\n'  #去除单引号，逗号，每行末尾追加换行符  .replace(' ','')
            output.write(s)
    ########################
    labels1 = get_labels("./CMUM1cy3.txt")
                #print(labels1,'labels1')
    train_dataset,train_features1 = format_data("./CMUM1cy3.txt")
    clf = DecisionTreeClassifier().fit(train_dataset, labels1)
    

    train_dataset1,train_features1 = format_data("./CMUM1cy2.txt")
    aa = clf.predict(train_dataset1)
    #print(aa,'aa')
    counters2 = [AccuracyCounter() for x in range(len(source_classes) + 1)]
    
    #label_tt = label_tt.tolist()
    alll2 = 0
    allc2 = 0
    for (each_predict_prob, each_label, label_all) in zip(all_indices1, label_tt, aa):
        alll2 = alll2+1
        if each_label in source_classes:
            #print(each_label,'each_label')
            counters2[each_label].Ntotal += 1.0
            #each_pred_id = np.argmax(each_predict_prob)

                    #if not outlier(each_target_share_weight, each_pred_id):
                        #counters[int(each_pred_id)].Npred += 1.0

            if label_all == '0.0' and each_predict_prob == each_label:
                allc2= allc2+1
                counters2[each_label].Ncorrect += 1.0
        else:
            counters2[-1].Ntotal += 1.0
            #each_pred_id = np.argmax(each_predict_prob)
            if label_all == '1.0':
                allc2 = allc2+1
                counters2[-1].Ncorrect += 1.0
    '''
    print('##############可信样本决策树############################')
    acc_tests2 = [x.reportAccuracy() for x in counters2 if not np.isnan(x.reportAccuracy())]
    print(acc_tests2,'acc_tests2')
    acc_test2 = torch.ones(1, 1) * np.mean(acc_tests2)

    print(f'test1 accuracy1 is {acc_test2.item()}')
    #print(allc2/alll2,'allc2/alll2')

    correct2 = [x.Ncorrect for x in counters2]
    amount2 = [x.Ntotal for x in counters2]
    common_acc2 = np.sum(correct2[0:-1]) / np.sum(amount2[0:-1])
    outlier_acc2 = correct2[-1] / amount2[-1]

    print('common_acc1={}, outlier_acc1={}'.format(common_acc2, outlier_acc2))
    bscore2 = 2 / (1 / common_acc2 + 1 / outlier_acc2)

    print('hscore1={}'.format(bscore2))
    '''
    if acc_test2 > best_acc2:
        best_acc2 = acc_test2
        bscore2_all = bscore2
    #print(best_acc2,'best_acc2')
    #print(bscore2_all,'bscore2_all')
    acc_all[2] = best_acc2
    acc_all[4] = bscore2_all

    ########################
    with open("./CMUM1cy4y.txt", "w") as output:
                #i = 0
        for i in range(9):
            s = str(acc_all[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
            s = s.replace("'",'').replace(',','') +'\n'  #去除单引号，逗号，每行末尾追加换行符  .replace(' ','')
            output.write(s)
    ###############################


    return best_acc,best_acc1,best_acc2,bscore1_all,bscore2_all,best_acc3,best_acc4,bscore3_all,bscore4_all


def test_class_inc(step, dataset_test, name, num_class, G, C, known_class):
    G.eval()
    C.eval()
    ## Known Score Calculation.
    correct = 0
    size = 0
    per_class_num = np.zeros((num_class))
    per_class_correct = np.zeros((num_class)).astype(np.float32)
    class_list = [i for i in range(num_class)]
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t, path_t = data[0], data[1], data[2]
            img_t, label_t = img_t.cuda(), label_t.cuda()
            feat = G(img_t)
            out_t = C(feat)
            out_t = F.softmax(out_t)
            pred = out_t.data.max(1)[1]
            correct += pred.eq(label_t.data).cpu().sum()
            pred = pred.cpu().numpy()
            k = label_t.data.size()[0]

            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
            size += k
            label_t = label_t.data.cpu().numpy()
            if batch_idx == 0:
                label_all = label_t
                pred_all = out_t.data.cpu().numpy()
            else:
                pred_all = np.r_[pred_all, out_t.data.cpu().numpy()]
                label_all = np.r_[label_all, label_t]
    per_class_acc = per_class_correct / per_class_num
    print(
        '\nTest set including unknown classes:  Accuracy: {}/{} ({:.0f}%)  '
        '({:.4f}%)\n'.format(
            correct, size,
            100. * correct / size, float(per_class_acc.mean())))
    close_p = 100. * float(correct) / float(size)
    output = [step, "closed", list(per_class_acc), float(per_class_acc.mean()),
              "acc known %s"%float(per_class_acc[:known_class].mean()),
              "acc novel %s"%float(per_class_acc[known_class:].mean()), "acc %s"%float(close_p)]
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=name, format="%(message)s")
    logger.setLevel(logging.INFO)
    logger.info(output)
    print(output)
    return float(per_class_acc[:known_class].mean()), float(per_class_acc[known_class:].mean())


def feat_get(step, G, C1, dataset_source, dataset_target, save_path):
    G.eval()
    C1.eval()

    for batch_idx, data in enumerate(dataset_source):
        if batch_idx == 500:
            break
        with torch.no_grad():
            img_s = data[0]
            label_s = data[1]
            img_s, label_s = Variable(img_s.cuda()), \
                             Variable(label_s.cuda())
            feat_s = G(img_s)
            if batch_idx == 0:
                feat_all_s = feat_s.data.cpu().numpy()
                label_all_s = label_s.data.cpu().numpy()
            else:
                feat_s = feat_s.data.cpu().numpy()
                label_s = label_s.data.cpu().numpy()
                feat_all_s = np.r_[feat_all_s, feat_s]
                label_all_s = np.r_[label_all_s, label_s]
    for batch_idx, data in enumerate(dataset_target):
        if batch_idx == 500:
            break
        with torch.no_grad():
            img_t = data[0]
            label_t = data[1]
            img_t, label_t = Variable(img_t.cuda()), \
                             Variable(label_t.cuda())
            feat_t = G(img_t)
            if batch_idx == 0:
                feat_all = feat_t.data.cpu().numpy()
                label_all = label_t.data.cpu().numpy()
            else:
                feat_t = feat_t.data.cpu().numpy()
                label_t = label_t.data.cpu().numpy()
                feat_all = np.r_[feat_all, feat_t]
                label_all = np.r_[label_all, label_t]
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "save_%s_target_feat.npy" % step), feat_all)
    np.save(os.path.join(save_path, "save_%s_source_feat.npy" % step), feat_all_s)
    np.save(os.path.join(save_path, "save_%s_target_label.npy" % step), label_all)
    np.save(os.path.join(save_path, "save_%s_source_label.npy" % step), label_all_s)
