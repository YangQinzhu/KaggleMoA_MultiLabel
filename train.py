import os
import gc
import random
import math
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import log_loss 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F

# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

#================================prepare=================
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICE'] = '1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_logger(filename='log'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

logger = get_logger()


def seed_everything(seed=42):
    random.seed(seed)  #控制生成的随机数等
    os.environ['PYTHONHASHSEED'] = str(seed) #python相关随机
    np.random.seed(seed)  
    torch.manual_seed(seed)  #torch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)

test_features = pd.read_csv('./moa_Data/test_features.csv')
train_features = pd.read_csv('./moa_Data/train_features.csv')
train_targets_scored = pd.read_csv('./moa_Data/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('./moa_Data/train_targets_nonscored.csv')
 
train = train_features.merge(train_targets_scored, on='sig_id') #根据sig id 进行合并

#记录训练的target每一列的标签名称，除sig id外
target_cols = [c for c in train_targets_scored.columns if c not in ['sig_id']] 

#在预测结果的标签列上加入 cp type
cols = target_cols + ['cp_type']

# train[cols].groupby('cp_type').sum().sum(1)  #依照cp_type组合，并集结所有的数据求和查看结果

print('train_features.shape, test_features.shape', train_features.shape, test_features.shape)
train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)  #去除控制扰动组
#同样，除去测试数据中的控制扰动组
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)  #调整索引序号
print('train.shape, test.shape', train.shape, test.shape)


folds = train.copy()
Fold = KFold(n_splits=5, shuffle=True, random_state=42)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[target_cols])):
    #loc： pandas的定位函数，可读取可写入，此处是写入
    folds.loc[val_index, 'fold'] = int(n) 
folds['fold'] = folds['fold'].astype(int)
print('folds.shape', folds.shape)  #加入fold，比train多出一列

#=====================================Data=====================================
class TrainDataset(Dataset):
    def __init__(self, df, num_features, cat_features, labels):
        self.cont_values = df[num_features].values
        self.cate_values = df[cat_features].values
        self.labels = labels
        
    def __len__(self):
        return len(self.cont_values)  #遍历所有的数据

    def __getitem__(self, idx):
        cont_x = torch.FloatTensor(self.cont_values[idx])
        cate_x = torch.FloatTensor(self.cate_values[idx])
        # cate_x = torch.LongTensor(self.cate_values[idx])
        label = torch.tensor(self.labels[idx]).float()
        
        #shape： cont_x:872;  cate_x: 2; label：206
        return cont_x, cate_x, label
    

class TestDataset(Dataset):
    def __init__(self, df, num_features, cat_features):
        self.cont_values = df[num_features].values
        self.cate_values = df[cat_features].values
        
    def __len__(self):
        return len(self.cont_values)

    def __getitem__(self, idx):
        cont_x = torch.FloatTensor(self.cont_values[idx])
        # cate_x = torch.LongTensor(self.cate_values[idx])
        cate_x = torch.FloatTensor(self.cate_values[idx])
        
        return cont_x, cate_x

# 单独分出 cp time、 cp dose
cat_features = ['cp_time', 'cp_dose']
num_features = [c for c in train.columns if train.dtypes[c] != 'object']
num_features = [c for c in num_features if c not in cat_features] 
num_features = [c for c in num_features if c not in target_cols] 
target = train[target_cols].values

def cate2num(df):
    df['cp_time'] = df['cp_time'].map({24: 0, 48: 1, 72: 2})
    df['cp_dose'] = df['cp_dose'].map({'D1': 3, 'D2': 4})
    return df

train = cate2num(train)  #将其中的time 和 dose转换为数字表示
test = cate2num(test)
#==========================================================================


#==============================train model===================
def train_fn(train_loader, model, optimizer, epoch, scheduler, device):
    
    losses = AverageMeter()

    model.train()

    for step, (cont_x, cate_x, y) in enumerate(train_loader):
        
        cont_x, cate_x, y = cont_x.to(device), cate_x.to(device), y.to(device)
        batch_size = cont_x.size(0)

        pred = model(cont_x, cate_x)  #包含两种数据feature的输入
        
        loss = nn.BCEWithLogitsLoss()(pred, y)  #二值交叉熵
        losses.update(loss.item(), batch_size) 

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
        
    return losses.avg


def validate_fn(valid_loader, model, device):
    
    losses = AverageMeter()

    model.eval()
    val_preds = []

    for step, (cont_x, cate_x, y) in enumerate(valid_loader):
        
        cont_x, cate_x, y = cont_x.to(device), cate_x.to(device), y.to(device)
        batch_size = cont_x.size(0)

        with torch.no_grad():
            pred = model(cont_x, cate_x)
            
        loss = nn.BCEWithLogitsLoss()(pred, y)
        losses.update(loss.item(), batch_size)

        #detach 返回从当前图中分离出来的变量
        val_preds.append(pred.sigmoid().detach().cpu().numpy())  #经过激活

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

    val_preds = np.concatenate(val_preds)
        
    return losses.avg, val_preds  #也返回一次测试的结果


def inference_fn(test_loader, model, device):  #对测试的模型推断函数

    model.eval()
    preds = []

    for step, (cont_x, cate_x) in enumerate(test_loader):

        cont_x,  cate_x = cont_x.to(device), cate_x.to(device)

        with torch.no_grad():
            pred = model(cont_x, cate_x)

        '''
        返回一个新的Variable，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，
        得到的这个Variable永远不需要计算其梯度，不具有grad.
        即使之后重新将它的requires_grad置为true,它也不会具有梯度grad.
        '''
        preds.append(pred.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def run_single_nn(cfg, train, test, folds, num_features, cat_features, target, device, fold_num=0, seed=42):
    
    # Set seed
    logger.info(f'Fold {fold_num}, Set seed {seed}')
    seed_everything(seed=seed)

    # loader
    trn_idx = folds[folds['fold'] != fold_num].index
    val_idx = folds[folds['fold'] == fold_num].index
    train_folds = train.loc[trn_idx].reset_index(drop=True)
    valid_folds = train.loc[val_idx].reset_index(drop=True)
    train_target = target[trn_idx]
    valid_target = target[val_idx]
    train_dataset = TrainDataset(train_folds, num_features, cat_features, train_target)
    valid_dataset = TrainDataset(valid_folds, num_features, cat_features, valid_target)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, 
                              num_workers=4, pin_memory=True, drop_last=False)

    # model
    # model = TabularNN(cfg)
    # model = Model_1(cfg)

    model = CNN_1D_pool(cfg)
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1e-2, epochs=cfg.epochs, steps_per_epoch=len(train_loader))

    # log
    log_df = pd.DataFrame(columns=(['EPOCH']+['TRAIN_LOSS']+['VALID_LOSS']) )

    # train & validate
    best_loss = np.inf
    for epoch in range(cfg.epochs):
        train_loss = train_fn(train_loader, model, optimizer, epoch, scheduler, device)
        valid_loss, val_preds = validate_fn(valid_loader, model, device) 
        log_row = {'EPOCH': epoch, 
                   'TRAIN_LOSS': train_loss,
                   'VALID_LOSS': valid_loss,
                  }
        log_df = log_df.append(pd.DataFrame(log_row, index=[0]), sort=False)
        #logger.info(log_df.tail(1))
        if valid_loss < best_loss:
            logger.info(f'epoch{epoch} save best model... {valid_loss}')
            best_loss = valid_loss
            oof = np.zeros((len(train), len(cfg.target_cols)))
            oof[val_idx] = val_preds   #保存每次最好的loss下的预测结果
            torch.save(model.state_dict(), f"fold{fold_num}_seed{seed}.pth")

    # predictions
    test_dataset = TestDataset(test, num_features, cat_features)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, 
                             num_workers=4, pin_memory=True)
    
    # model = TabularNN(cfg)
    model = CNN_1D_pool(cfg)
    # model = Model_1(cfg)
    
    model.load_state_dict(torch.load(f"fold{fold_num}_seed{seed}.pth"))
    model.to(device)
    predictions = inference_fn(test_loader, model, device)  #推断，是对测试数据的推断
    
    # del
    torch.cuda.empty_cache()  #清空cuda缓存

    return oof, predictions, best_loss


def run_kfold_nn(cfg, train, test, folds, num_features, cat_features, target, device, n_fold=5, seed=42):

    oof = np.zeros((len(train), len(cfg.target_cols)))
    predictions = np.zeros((len(test), len(cfg.target_cols)))

    valBestList_KFold = []
    for _fold in range(n_fold):
        logger.info("Fold {}".format(_fold))
        _oof, _predictions, best_loss= run_single_nn(cfg,
                                           train,
                                           test,
                                           folds,
                                           num_features, 
                                           cat_features,
                                           target, 
                                           device,
                                           fold_num=_fold,
                                           seed=seed)
        oof += _oof
        predictions += _predictions / n_fold
        valBestList_KFold.append(best_loss)

    score = 0
    for i in range(target.shape[1]):
        _score = log_loss(target[:,i], oof[:,i])
        score += _score / target.shape[1]
    logger.info(f"CV score: {score}")   #一次K折验证下的均值
    
    #返回最好的oof、在测试集上的测试结果，交叉验证的每次结果的记录
    return oof, predictions, valBestList_KFold

#================================model 
class CFG:  #模型配置参数
    max_grad_norm=1000 #梯度裁剪
    gradient_accumulation_steps=1
    hidden_size=512 
    dropout=0.5
    lr=1e-2
    weight_decay=1e-6
    batch_size=32
    epochs=50  #最大训练回合
    #total_cate_size=5
    #emb_size=4
    num_features=num_features #数据复制
    cat_features=cat_features
    target_cols=target_cols

# class TabularNN(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
        # self.mlp = nn.Sequential(
        #                   nn.Linear(len(cfg.num_features), cfg.hidden_size),
        #                   nn.BatchNorm1d(cfg.hidden_size),
        #                   nn.Dropout(cfg.dropout),
        #                   nn.PReLU(),
        #                   nn.Linear(cfg.hidden_size, cfg.hidden_size),
        #                   nn.BatchNorm1d(cfg.hidden_size),
        #                   nn.Dropout(cfg.dropout),
        #                   nn.PReLU(),
        #                   nn.Linear(cfg.hidden_size, len(cfg.target_cols)),
                        #   )

    # def forward(self, cont_x, cate_x): 
    #     # no use of cate_x yet
    #     x = self.mlp(cont_x)
    #     return x


# class TabularNN(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()

#         self.mlp = nn.Sequential(
#                           nn.Linear(len(cfg.cat_features)+len(cfg.num_features), cfg.hidden_size),
#                           nn.BatchNorm1d(cfg.hidden_size),
#                           nn.Dropout(cfg.dropout),
#                           nn.PReLU(),
#                           nn.Linear(cfg.hidden_size, cfg.hidden_size),
#                           nn.BatchNorm1d(cfg.hidden_size),
#                           nn.Dropout(cfg.dropout),
#                           nn.PReLU(),
#                           nn.Linear(cfg.hidden_size, len(cfg.target_cols)),
#                           )

#     def forward(self, cont_x, cate_x): 
#         x = self.mlp(torch.cat((cont_x, cate_x), axis=1))  #（B， N）
#         return x

# class CNN_1D(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()

#         self.cnn = nn.Sequential(
#                           nn.Conv1d(1, 4, kernel_size=3, stride=1, padding=0),
#                           nn.BatchNorm1d(4),
#                           nn.Dropout(cfg.dropout),
#                           nn.PReLU(),
#                           nn.Conv1d(4, 8, kernel_size=3, stride=1, padding=0),
#                           nn.BatchNorm1d(8),
#                           nn.Dropout(cfg.dropout),
#                           nn.PReLU(),
#                           )
#         #每卷积一次减少2
#         self.linear = nn.Linear(8*(len(cfg.cat_features)+len(cfg.num_features)-2*2), len(cfg.target_cols))

#     def forward(self, cont_x, cate_x): 
#         inTensor = torch.cat((cont_x, cate_x), axis=1)
#         inTensor = torch.unsqueeze(inTensor, dim=1)
#         x = self.cnn(inTensor)  #（B， N）（32， 874）
#         x = torch.flatten(x,start_dim=1)
#         x = self.linear(x)
#         return x

# class Model_1(nn.Module):
#     def __init__(self,cfg, input_size=0,output_size=0,hidden_size=0):
#         super(Model_1,self).__init__()
#         input_size = len(cfg.cat_features)+len(cfg.num_features)
#         output_size = len(cfg.target_cols)
#         hidden_size = 1024
#         self.batch_norm1 = nn.BatchNorm1d(input_size)
#         self.dropout1 = nn.Dropout(0.5)
#         self.linear1 = nn.utils.weight_norm(nn.Linear(input_size,hidden_size))
        
#         self.batch_norm2 = nn.BatchNorm1d(hidden_size)
#         self.dropout2 = nn.Dropout(0.6)
#         self.linear2 = nn.utils.weight_norm(nn.Linear(hidden_size,hidden_size))
        
#         self.batch_norm3 = nn.BatchNorm1d(hidden_size)
#         self.dropout3 = nn.Dropout(0.6)
#         self.linear3 = nn.utils.weight_norm(nn.Linear(hidden_size,output_size))
        
#     def forward(self, cont_x, cate_x):
#         xb = torch.cat((cont_x, cate_x), axis=1)
#         x = self.batch_norm1(xb)
#         x = self.dropout1(x)
#         x = F.leaky_relu(self.linear1(x))
        
#         x = self.batch_norm2(x)
#         x = self.dropout2(x)
#         x = F.leaky_relu(self.linear2(x))
        
#         x = self.batch_norm3(x)
#         x = self.dropout3(x)
#         return self.linear3(x)


# class CNN_1D(nn.Module): #无dropout
#     def __init__(self, cfg):
#         super().__init__()

#         self.cnn = nn.Sequential(
#                           nn.Conv1d(1, 4, kernel_size=3, stride=1, padding=0),
#                           nn.BatchNorm1d(4),
#                           nn.Dropout(cfg.dropout),
#                           nn.PReLU(),
#                           nn.Conv1d(4, 4, kernel_size=3, stride=1, padding=0),
#                           nn.BatchNorm1d(4),
#                           nn.Dropout(cfg.dropout),
#                           nn.PReLU(),
#                           )
#         #每卷积一次减少2
#         self.linear = nn.Linear(4*(len(cfg.cat_features)+len(cfg.num_features)-2*2), len(cfg.target_cols))

#     def forward(self, cont_x, cate_x): 
#         inTensor = torch.cat((cont_x, cate_x), axis=1)
#         inTensor = torch.unsqueeze(inTensor, dim=1)
#         x = self.cnn(inTensor)  #（B， N）（32， 874）
#         x = torch.flatten(x,start_dim=1)
#         x = self.linear(x)
#         return x

class CNN_1D_pool(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cnn = nn.Sequential(
                        #输入输出通道数目,卷积核为3，步长为1，padding为0，最终会造成卷积后的某一维长度缩小3-1=2
                          nn.Conv1d(1, 4, kernel_size=3, stride=1, padding=0), 
                          nn.BatchNorm1d(4), #channel
                          nn.Dropout(cfg.dropout),
                          nn.PReLU(),
                          nn.MaxPool1d(2, stride=2),  #最大池化，但此处不影响通道数目
                          nn.Conv1d(4, 8, kernel_size=3, stride=1, padding=0),  #输入输出通道数目
                          nn.BatchNorm1d(8),
                          nn.Dropout(cfg.dropout),
                          nn.PReLU(),
                          nn.MaxPool1d(2, stride=2), #最大池化，但此处不影响通道数目
                          )
        #每卷积一次减少2
        self.linear = nn.Linear(8*(((len(cfg.cat_features)+len(cfg.num_features)-2) //2) -2) //2, len(cfg.target_cols))  #卷积了一次，所以减少了2，/2代表最大池化的效果

    def forward(self, cont_x, cate_x): 
        inTensor = torch.cat((cont_x, cate_x), axis=1)
        inTensor = torch.unsqueeze(inTensor, dim=1)
        x = self.cnn(inTensor)  #（B， N）（32， 874）
        x = torch.flatten(x,start_dim=1)
        x = self.linear(x)
        return x


#==================================================

#=====================post process=================
# Seed Averaging for solid result
oof = np.zeros((len(train), len(CFG.target_cols))) #21948, 206
predictions = np.zeros((len(test), len(CFG.target_cols)))

# SEED = [0, 1, 2]  #random seed 设置多次随机种子进行多次验证
SEED = [0]  #random seed， 
valBestList_Seed = []  
# SEED = [0] #先跑一次查看结果
for seed in SEED:  #在K折交叉验证中，设置不同的随机种子，进行均值计算
    _oof, _predictions, valBestList_KFold = run_kfold_nn(CFG, 
                                      train, test, folds, 
                                      num_features, cat_features, target,
                                      device,
                                      n_fold=5, seed=seed) 

    #_oof:(21948, 206); pred shape:(3624, 206)
    # _oof, _predictions = run_single_nn(CFG, train, test, folds, num_features, cat_features, target, device)
    oof += _oof / len(SEED)     
    predictions += _predictions / len(SEED)
    valBestList_Seed.append(valBestList_KFold)
    # print(np.mean(valBestList_KFold))

logger.info(f'val best loss in every different seed: \n {valBestList_Seed}')

score = 0
for i in range(target.shape[1]):
    _score = log_loss(target[:,i], oof[:,i])  #每一个类别计算交叉熵， oof是在验证集上的预测结果,五个验证集数据一起计算
    score += _score / target.shape[1]       #再根据预测数据的数目求出均值，作为最终的分数

logger.info('CNN 1D, three layer, k3, k3, k1')
logger.info(f"Different Seed {len(SEED)} Averaged CV score: {score}")

#20201224 没有cp dose、time， 跑了1次，求均值
# Different Seed 1 Averaged CV score: 0.01627191708850361， 

#20201225 没有cp dose、time， 跑了3次，求均值 ；特征 872维度
# Different Seed 3 Averaged CV score: 0.01611340856627774

#20201224 包含cp dose、time， 跑了3次，求均值 ；特征 874维度
# Different Seed 3 Averaged CV score: 0.016123690085135107

'''
channel 1-4， 4-8， 8-16
CV score: 0.01776334404073402
val best loss in every different seed: 
 [[0.017765716945960405, 0.017577624827589424, 0.017677119301296857, 0.01842130241904125, 0.017634134431742853], [0.017645368217183002, 0.01744176864471278, 0.01761238342554124, 0.018193452008107604, 0.017705602421134364], [0.017732392992306677, 0.01749938730158947, 0.017752708636377016, 0.018232634458474443, 0.017599655033909777]]
CNN 1D, three layer, k3, k3, k1
Different Seed 3 Averaged CV score: 0.01705334438011983


channel 1-4， 4-4,4-8
CV score: 0.01773243784628315
val best loss in every different seed: 
 [[0.017559613498110232, 0.01763888615987567, 0.01755245705145774, 0.018326413148017223, 0.017485921297669682], [0.017533290930567543, 0.017558397260199646, 0.01770993653888515, 0.0183117023976229, 0.017697293751873712], [0.01743227100436807, 0.017609878250128316, 0.017657032600924338, 0.01829810268471625, 0.017665035709463726]]
CNN 1D, three layer, k3, k3, k1
Different Seed 3 Averaged CV score: 0.01705079583627751

@20201228
channel 1-4， 4-8

channel 1-4 ,pool 2d, 4-8, pool 2d, 全连接层



'''

'''
channel 1-4， 4-4， 4-8

'''

# nohup python train.py >> log1dCNN.log 2>&1 &  [1] 22519