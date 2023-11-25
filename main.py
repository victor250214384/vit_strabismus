import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder
import pandas as pd
import random
import time
import torchvision
import torch.nn as nn
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
import os
from loguru import logger
import glob
from mymodel.vit_model import VIT_B16_224, VIT_16_256
from mymodel.chmodel import NET
from mymodel.InceptionResNetV2 import inceptionresnetv2
from mymodel.ViTModel import Vit_256_16
import timm

ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICE'] = "0,1,2,3"
device_ids = [0,1,2,3]  

# torch.cuda.set_device(0)
modellr = 2e-4#学习率
BATCH_SIZE = 256
EPOCHS = 8
# DEVICE = torch.device('cuda:[0,1,2,3]' if torch.cuda.is_available() else 'cpu')
class_name = ['    apple',' other_fruit',]
expand_times = 3
image_size = 224
# height = int(image_size / (3686/850))+1
# height = 48
# print(height)
# print(int((image_size-height)/2) * 2 + height)

train_transforms = transforms.Compose(
    [
        transforms.Resize((224,224)),
        # transforms.Resize((height,image_size)),
        # transforms.Pad([0,int((image_size-height)/2)],fill=(0,0,0),padding_mode='constant'),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomRotation((-5, 5)),
        transforms.ColorJitter(contrast=[0.8,1.2]),
        transforms.RandomAutocontrast(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ]
)
test_transforms = transforms.Compose(
    [ 
        transforms.Resize((224,224)),
        # transforms.Resize((height,image_size)),
        # transforms.Pad([0, int((image_size-height)/2)],fill=(0,0,0),padding_mode='constant'),
        # transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ]
)

def Open(x):
    I = Image.open(x).convert('RGB')
#     I = I.convert('L')
    return I

def cut_image(img):
    width, height = img.size
    limg = img.crop((0, 0, width/2, height)) 
    rimg = img.crop((width/2, 0, width, height)) 
    return limg, rimg

class MyDataset(Dataset):
    def __init__(self, data_list, transforms=None):
        self.data_list = data_list
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, label = self.data_list[idx]

        # 读取图像
        img = Image.open(image_path).convert('RGB')

        # 如果定义了transforms，则对图像进行预处理
        if self.transforms is not None:
            img = self.transforms(img)
        # print(img.shape)
        return img, label
    
# train_set = DatasetFolder("/disk/user/lyf/eyes/split_data/train", loader=Open, extensions=("jpg","png"), transform=train_transforms)
# # train_set2 = DatasetFolder("/disk/user/lyf/eyes/split_data/normal", loader=Open, extensions=("jpg","png"), transform=test_transforms)
# # train_set = train_set+ train_set2
def create_dataset_lists(root_folder):
    image_paths = []  # 存储图片相对路径的列表
    folder_indices = []  # 存储文件夹索引的列表

    # 获取文件夹下所有子文件夹
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

    # 遍历每个子文件夹
    for folder_index, folder_path in enumerate(subfolders):
        # 获取该子文件夹下所有图片文件相对路径
        image_names = [f.name for f in os.scandir(folder_path) if f.is_file() and f.name.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_paths.extend([os.path.join(folder_path, name) for name in image_names])
        folder_indices.extend([folder_index] * len(image_names))

    return image_paths, folder_indices

root_folder = "apple_data/apple_train"

image_paths, folder_indices = create_dataset_lists(root_folder)
test_imgpath, test_foldidx  = create_dataset_lists("apple_data/apple_train")
testdata = [(test_imgpath[i],test_foldidx[i]) for i in range(len(test_imgpath))]
test_dataset = MyDataset(testdata, test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)

# from timm.models import inception_resnet_v2
# model = inception_resnet_v2(pretrained=True ,num_classes = 2)

# from dencenet import densenet121
# model = densenet121(num_classes = 2)
# pretrained_model_dict = torch.load('/disk/user/lyf/vit/densenet121.pth')

# from swin_transformer import swin_tiny_patch4_window7_224 as create_model
# model = create_model(num_classes=2).to(device_ids[0])
# print(model)
# keys = []
# pretrained_model_dict = torch.load('/disk/user/lyf/vit/swin_tiny_patch4_window7_224.pth')
# for k,v in pretrained_model_dict.items():
#     if k.startswith('head'):    #将‘conv_cls’开头的key过滤掉，这里是要去除的层的key
#         continue
#     keys.append(k)
# new_dict = {k:pretrained_model_dict[k] for k in keys}
# model.load_state_dict(new_dict, strict=False)

# model = VIT_B16_224(num_classes = 2)
# pretrained_model_dict = torch.load('/disk/user/lyf/vit/mymodel/imagenet_vit.pth')
# # pretrained_model_dict = torch.load('/disk/user/lyf/vit/kfold_model/m50.pth')
# keys = []
# for k,v in pretrained_model_dict.items():
#     if k.startswith('mlp_head'):    #将‘conv_cls’开头的key过滤掉，这里是要去除的层的key
#         continue
#     keys.append(k)
# new_dict = {k:pretrained_model_dict[k] for k in keys}
# model.load_state_dict(new_dict, strict=False)


# model = torch.nn.DataParallel(model, device_ids=device_ids)
# model = model.cuda(device = device_ids[0])
# # if isinstance(model, torch.nn.DataParallel):
# #    model = model.module

# # model = model.cuda(device=device_ids[0])
# # model.eval()

class focal_loss(nn.Module):
    def __init__(self, alpha=0.4, gamma=2, num_classes = 2, size_average=True):
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1  
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1) 
        preds_logsoft = torch.log(preds_softmax)
        
        #focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1)) 
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) 

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
    
# criterion=focal_loss()
# criterion = nn.CrossEntropyLoss() #交叉熵函数
# # criterion = LDAMLoss(cls_num_list=train_sum, max_m=0.5, s=30).cuda(0)
# optimizer = optim.Adam(model.parameters(), lr=modellr, weight_decay=0.0001) #Adam优化器
# cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=30, eta_min=1e-9)
# optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 20))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True  # 为了确定算法，保证得到一样的结果。
     torch.backends.cudnn.enabled = True  # 使用非确定性算法
     torch.backends.cudnn.benchmark = True  # 是否自动加速。
# 设置随机数种子

suffix = "2023021701_tbar1"
ckpt_path = None # "ckpts/lhy/best_9956.pth"
    
ckpt_folder = os.path.join("ckpts", suffix)
log_path = os.path.join("logs", "%s.log"%(suffix))
    
logger.add(log_path, level="INFO")
    
if not os.path.exists(ckpt_folder):
    os.makedirs(ckpt_folder)

best_acc = 0
Train_loss, Test_loss, Train_acc, Test_acc = [],[],[],[]

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
def print_confusion_matrix(y_true, y_pred, type, epoch_id):
    C = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 计算每个单元格的数据量和百分比，并添加到混淆矩阵中
    total_samples = np.sum(C,axis=1)
    cm_with_stats = np.zeros_like(C, dtype=float)
    cm_annotations = []

    labels = ['Normal', 'Strabismus']
    for i in range(C.shape[0]):
        row_stats = []
        for j in range(C.shape[1]):
            count = C[i, j]
            cm_with_stats[i, j] = count / total_samples[i]  # 存储百分比而不是字符串
            row_stats.append(f'{count} ({(count / total_samples[i]) * 100:.2f}%)')
        cm_annotations.append(row_stats)
    # 转换为NumPy数组
    cm_annotations = np.array(cm_annotations)
    sns.heatmap(cm_with_stats, annot=cm_annotations, fmt='', cmap=plt.cm.Reds, xticklabels=labels, yticklabels=labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix with Counts and Percentages')
    f = plt.gcf()
    f.savefig('vit/matrix.jpg'.format(type,epoch_id))
    f.clear()  #释放内存


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import itertools
def plotTheCurve(fold):
    for train_loss in Train_loss:
        plt.plot([i for i in range(len(train_loss))],train_loss,'',label="train_loss")
    plt.title('loss')
    plt.legend(loc='upper right', labels=['fold 1','fold 2','fold 3','fold 4','fold 5'])
    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    f = plt.gcf()
    f.savefig('/disk/user/lyf/vit/resultinceptionresnet/LossCurve/train_{}.jpg'.format(fold))
    f.clear()  #释放内存
    plt.show()
    
    for test_loss in Test_loss:
        plt.plot([i for i in range(len(test_loss))],test_loss,'',label="train_loss")
    plt.title('loss')
    plt.legend(loc='upper right', labels=['fold 1','fold 2','fold 3','fold 4','fold 5'])
    plt.xlabel('epoch')
    plt.ylabel('test_loss')
    f = plt.gcf()
    f.savefig('/disk/user/lyf/vit/resultinceptionresnet/LossCurve/test_{}.jpg'.format(fold))
    f.clear()  #释放内存
    plt.show()
    
    # plt.figure(figsize=(8,6))
    for train_acc in Train_acc:
        plt.plot([i for i in range(len(train_acc))],train_acc,'',label="acc")
    plt.title('acc')
    plt.legend(loc='lower right', labels=['fold 1','fold 2','fold 3','fold 4','fold 5'])
    plt.xlabel('epoch')
    plt.ylabel('train acc')
    # plt.grid(len(Train_acc[0]))
    f = plt.gcf()
    f.savefig('/disk/user/lyf/vit/resultinceptionresnet/AccCurve/train_{}.jpg'.format(fold))
    f.clear()  #释放内存
    plt.show()
    
    for test_acc in Test_acc:
        plt.plot([i for i in range(len(test_acc))],test_acc,'',label="acc")
    plt.title('acc')
    plt.legend(loc='lower right', labels=['fold 1','fold 2','fold 3','fold 4','fold 5'])
    plt.xlabel('epoch')
    plt.ylabel('test acc')
    f = plt.gcf()
    f.savefig('/disk/user/lyf/vit/resultinceptionresnet/AccCurve/test_{}.jpg'.format(fold))
    f.clear()  #释放内存
    plt.show()

def print_roc_curves(type,epoch_id, true_label, preds):
    n_classes = 2
    true_label = np.array(true_label)
    preds = np.array(preds)
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        # print(i)
        fpr[i], tpr[i], _ = roc_curve(true_label[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(true_label.ravel(), preds.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    lw=2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)
    
    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('multi-calss ROC')
    plt.legend(loc='lower right', labels=['fold 1','fold 2','fold 3','fold 4','fold 5'])
    f = plt.gcf()
    f.savefig('/disk/user/lyf/vit/resultinceptionresnet/RocCurve/{}{}.jpg'.format(type,epoch_id))
    f.clear()  #释放内存
    plt.show()
    return roc_auc["macro"]

def print_roc_curve_final(type, epoch_id, true_labels, predss):
    n_classes = 2
    lw=2
    plt.figure()
    # line_styles = ['-','-','-','-','-','-']
    # line_width = [2,2,2,2,2,4]
    colors = ['blue','m','green','purple','orange','brown']
    fold = 0
    AUCs = []
    tpr_mean = 0
    common_fpr = np.linspace(0, 1, 150)  # 假设我们选择100个点作为公共维度
    for true_label, preds in zip(true_labels, predss):
        fold = fold + 1
        # 计算每一类的ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        true_label = np.array(true_label)
        preds = np.array(preds)
        # print(true_label.shape)
        # print(preds.shape)
        for i in range(n_classes):
            # print(i)
            fpr[i], tpr[i], _ = roc_curve(true_label[:, i], preds[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # fpr["micro"], tpr["micro"], _ = roc_curve(true_label.ravel(), preds.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        # print(all_fpr)
        # print(mean_tpr)
        interp_tpr_fold = np.interp(common_fpr, all_fpr, mean_tpr)
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        # print(all_fpr.shape)
        # print(mean_tpr.shape)
        # Plot all ROC curves
        # plt.plot(fpr["micro"], tpr["micro"],
        #         label='micro-average ROC curve (area = {0:0.2f})'
        #             ''.format(roc_auc["micro"]),
        #         color='deeppink', linestyle=':', linewidth=4)
        if fold < 6:
            if fold == 1:
                tpr_mean = interp_tpr_fold
            else:
                tpr_mean = tpr_mean + interp_tpr_fold
            AUCs.append(roc_auc["macro"])
            plt.plot(fpr["macro"], tpr["macro"],
                    label='ROC fold {0} (AUC = {1:0.2f})'.format(fold, roc_auc["macro"]),
                    color=colors[fold-1], linestyle='-', linewidth=2)
        else:
            plt.plot(fpr["macro"], tpr["macro"],
                    label='ROC test (AUC = {0:0.2f})'
                        ''.format(roc_auc["macro"]),
                    color=colors[fold-1], linestyle='-', linewidth=4)
        # colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
        # for i, color in zip(range(n_classes), colors):
        #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
        #             label='ROC curve of class {0} (area = {1:0.2f})'
        #             ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw, color = 'red', label='Chance')
    # fpr_mean = fpr_mean / (len(true_labels)-1)
    tpr_mean = tpr_mean / (len(true_labels)-1)
    mean_value = np.mean(AUCs)
    # 计算标准差
    std_deviation = np.std(AUCs)
    # 保留两位小数
    mean_value = round(mean_value, 2)
    std_deviation = round(std_deviation, 2)
    print(std_deviation)
    # print(common_fpr)
    # print(tpr_mean)
    plt.plot(common_fpr, tpr_mean,
                label='Mean ROC (AUC = {0:0.2f} ± {1:0.2f})'
                    ''.format(mean_value, std_deviation),
                color='c', linestyle='-', linewidth=4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right",fontsize='small')
    f = plt.gcf()
    f.savefig('/disk/user/lyf/vit/resultinceptionresnet/RocCurve/final_{}{}.jpg'.format(type,epoch_id))
    f.clear()  #释放内存

@torch.no_grad()
def Test(epoch, model, test_data_loader, criterion):
    print('Test')
    model.eval()
    tk1 = tqdm(test_data_loader, total=int(len(test_data_loader)))
    acc_num, sum_num, loss_sum = 0, 0, 0
    class_num = [0,0]
    class_sum = [0,0]
    total_labels = total_outputs = total_scores = []
    One_hot_labels = []
    TP = FP = FN = TN = 0
    for bi, (inputs,labels) in enumerate(tk1):
        inputs, labels = inputs.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0])
        outputs = model(inputs)
        # print(outputs.shape)
        # print(labels.shape)
        total_outputs = np.append(total_outputs, outputs.argmax(-1).cpu().numpy())
        total_labels = np.append(total_labels, labels.cpu().numpy())
        One_hot_labels = np.append(One_hot_labels, [int(x) for x in labels])
        # print(outputs.shape)
        for row in F.softmax(outputs, dim=1).detach().cpu():
            total_scores.append(row.tolist())
        loss = criterion(outputs, labels)
        for i in range(BATCH_SIZE): 
            if i == len(outputs.argmax(-1)):
                break
            if outputs.argmax(-1)[i].item() == labels[i].item():
                class_num[labels[i].item()] += 1
            class_sum[labels[i].item()] += 1
        acc_num += (labels==outputs.argmax(-1)).sum()
        sum_num += labels.size(0)
        loss_sum += loss.item() * inputs.size(0)
        # break
    # print(One_hot_labels.shape)
    One_hot_labels = F.one_hot(torch.from_numpy(np.array(One_hot_labels).astype(int))).cpu().tolist()
    acc = acc_num / sum_num
    class_acc = np.array(class_num) / np.array(class_sum)
    for i in range(2):
        logger.info('Test class_{}: {}  ACC: {:.4f}  {}'.format(i, class_sum[i], class_acc[i], class_name[i][:-4]))
        print('Test class_{}: {}  ACC: {:.4f}  {}'.format(i, class_sum[i], class_acc[i], class_name[i][:-4]))
    print('Test Acc: {:.4f}'.format(acc))
    logger.info('Test Acc: {:.4f}'.format(acc))
    TN, FP, FN, TP = confusion_matrix(total_labels,  total_outputs, labels=[0, 1]).ravel()
    Precision = TP / (TP + FP)
    Recall = TP / (TP+FN)
    Specificity = TN / (TN + FP)
    Sensitivity = TP / (TP + FN)
    F1 =  2 * (Precision * Recall) /  (Precision + Recall)
    print('Precision: {}'.format(Precision) )
    logger.info('Precision: {}'.format(Precision) )
    print('Recall:    {}'.format(Recall))
    logger.info('Recall:    {}'.format(Recall))
    print('Specificity: {}'.format(Specificity))
    logger.info('Specificity:  {}'.format(Specificity))
    print('Sensitivity: {}'.format(Sensitivity))
    logger.info('Sensitivity:  {}'.format(Sensitivity))
    print('F1_score:  {}'.format(F1))
    logger.info('F1_score:  {}'.format(F1))
    # fpr,tpr,thre = roc_curve(One_hot_labels, scores)
    # Auc = auc(fpr,tpr)
    if  epoch % 10 == 0:       
        print('------------confusion_matrix------------- ')
    #     print(total_outputs.shape)
        print_confusion_matrix(total_labels , total_outputs, 'test', epoch)
        Auc = print_roc_curves('test_',epoch, One_hot_labels, total_scores)
        print('Auc:  {}'.format(Auc))
        logger.info('Auc:  {}'.format(Auc))
    global best_acc
    test_loss = loss_sum / test_data_loader.batch_size
    test_acc = acc.cpu()
    return test_loss, test_acc

# from sklearn.model_selection import  StratifiedKFold, KFold
# skf = KFold(n_splits=5,shuffle=True) #5折
import warnings
warnings.filterwarnings("ignore")

def train_model(model, train_loader, val_loader, criterion, optimizer, cosine_schedule):
    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    total_labels = total_outputs = []
    for epoch in range(0, EPOCHS+1):
        One_hot_label, total_scores = [],[]
        print('Epoch {}/{}'.format(epoch, EPOCHS))
        logger.info('Epoch {}/{}'.format(epoch, EPOCHS))
        print('-' * 30)
        logger.info('-' * 30)
        adjust_learning_rate(optimizer, epoch)
    #     print("lr:", scheduler.get_lr()[0])
        model.train()
        running_loss = 0.0
        tk0 = tqdm(train_loader, total=int(len(train_loader)))
        counter = 0
        acc_num, sum_num = 0, 0
        class_num = [0,0]
        class_sum = [0,0]
        for bi, (inputs, labels) in enumerate(tk0):
            inputs, labels = inputs.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0])
            # linputs, rinputs, labels = linputs.cuda(device=device_ids[0]), rinputs.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0])
            optimizer.zero_grad()
            # with torch.set_grad_enabled(True):
            # outputs = model(inputs)
            outputs = model(inputs)
            total_outputs = np.append(total_outputs, outputs.argmax(-1).cpu().numpy())
            # print(total_outputs.shape())
            total_labels = np.append(total_labels, labels.cpu().numpy())
            # One_hot_labels = torch.cat((One_hot_labels, labels), dim=0)
            One_hot_label = np.append(One_hot_label, [int(x) for x in labels])
            # print(F.softmax(outputs, dim=1).detach().cpu().numpy())
            # print(F.softmax(outputs, dim=1).detach().cpu())
            for row in F.softmax(outputs, dim=1).detach().cpu():
                total_scores.append(row.tolist())
            # print(len(total_scores))
            # print(scores)
            # outputs = model(linputs, rinputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            for i in range(BATCH_SIZE): 
                if i == len(outputs.argmax(-1)):
                    break
                if outputs.argmax(-1)[i].item() == labels[i].item():
                    class_num[labels[i].item()] += 1
                class_sum[labels[i].item()] += 1
            acc_num += (labels==outputs.argmax(-1)).sum()
            sum_num += labels.size(0)
            counter += 1
            tk0.set_postfix(loss=(running_loss / (counter * train_loader.batch_size)))
        One_hot_labels = F.one_hot(torch.from_numpy(np.array(One_hot_label).astype(int))).cpu().tolist()
        # print(One_hot_labels)                  
        class_acc = np.array(class_num) / np.array(class_sum)
        epoch_loss = running_loss / len(train_loader) / train_loader.batch_size
        acc = acc_num / sum_num
        for i in range(2):
            logger.info('Train class_{}: {}  ACC: {:.4f}  {}'.format(i, class_sum[i], class_acc[i], class_name[i]))
            print('Train class_{}: {}  ACC: {:.4f}  {}'.format(i, class_sum[i], class_acc[i], class_name[i]))
        logger.info('Training Loss: {:.4f}, Training Acc: {:.4f}'.format(epoch_loss, acc))
        print('Training Loss: {:.4f}, Training Acc: {:.4f}'.format(epoch_loss, acc))
        cosine_schedule.step()
        TN, FP, FN, TP = confusion_matrix(total_labels,  total_outputs, labels=[0, 1]).ravel()
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        Sensitivity = TP / (TP + FN)
        F1 =  2 * (Precision * Recall) /  (Precision + Recall)
        print('Precision: {}'.format(Precision) )
        logger.info('Precision: {}'.format(Precision) )
        print('Recall:    {}'.format(Recall))
        logger.info('Recall:    {}'.format(Recall))
        print('Specificity: {}'.format(Specificity))
        logger.info('Specificity:  {}'.format(Specificity))
        print('Sensitivity: {}'.format(Sensitivity))
        logger.info('Sensitivity:  {}'.format(Sensitivity))
        print('F1_score:  {}'.format(F1))
        logger.info('F1_score:  {}'.format(F1))
        # fpr,tpr,thre = roc_curve(One_hot_labels, scores)
        # Auc = auc(fpr,tpr)
        print(type(One_hot_labels[0]))
        print(len(One_hot_labels))
        print(type(total_scores[0]))
        print(len(total_scores))
        print(len(total_scores[0]))
        if  epoch % 10 == 0:       
            print('------------confusion_matrix------------- ')
        #     print(total_outputs.shape)
            print_confusion_matrix(total_labels , total_outputs, 'train', epoch)
            Auc = print_roc_curves('train_',epoch, One_hot_labels, total_scores)
            print('Auc:  {}'.format(Auc))
            logger.info('Auc:  {}'.format(Auc))
        # Valid()
        loss,acc_test = Test(epoch, model, val_loader, criterion)
        test_loss.append(loss)
        test_acc.append(acc_test)
        train_loss.append(epoch_loss)
        train_acc.append(acc.cpu())
        if  epoch % 1 == 0:
            torch.save(model.state_dict(), "/disk/user/lyf/vit/inception_model/inceptionresnet_{}.pth".format(epoch))
    Train_loss.append(train_loss)
    Train_acc.append(train_acc)
    Test_loss.append(test_loss)
    Test_acc.append(test_acc)
    # print(Train_loss,Train_acc, Test_loss,Test_acc)

from sklearn.model_selection import KFold, StratifiedKFold
def k_fold_cross_validation(datapath, labels, num_folds=5):
    # kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_accuracies , test_fold_accuracies, Scores, TestScores, Labels, TestLabels= [],[], [],[],[],[]
    for fold, (train_indices, val_indices) in enumerate(kf.split(datapath, labels)):
        setup_seed(20)
        print(f"Fold {fold+1}/{num_folds}")
        # 划分训练集和验证集
        traindata = [(datapath[i],labels[i]) for i in train_indices]
        expanddata = []
        for item in traindata:
            image_name, label = item
            expanddata.append(item)
            if label == 1:
                # 将该元组扩充五倍
                for _ in range(expand_times):
                    expanddata.append((image_name, label))
        valdata = [(datapath[i],labels[i]) for i in val_indices]
        train_dataset = MyDataset(expanddata, train_transforms)
        val_dataset = MyDataset(valdata, test_transforms)
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)
        
        # 初始化模型、损失函数和优化器
        # model = timm.create_model('inception_v3', pretrained=None)
        # model.load_state_dict(torch.load('vit/mymodel/inception_v3.pth'), strict=False)
        # model = VIT_B16_224(num_classes = 2)
        # pretrained_model_dict = torch.load('vit/mymodel/imagenet_vit.pth')
        # keys = []
        # for k,v in pretrained_model_dict.items():
        #     if k.startswith('mlp_head'):    #将‘conv_cls’开头的key过滤掉，这里是要去除的层的key
        #         continue
        #     keys.append(k)
        # new_dict = {k:pretrained_model_dict[k] for k in keys}
        # model.load_state_dict(new_dict, strict=False)
        
        # model = NET()
        # model.load_state_dict(torch.load('vit/Net.pth'), strict=True)
        model = inceptionresnetv2(num_classes=1000, pretrained=None)
        model.load_state_dict(torch.load('vit/mymodel/inception_resnet_v2.pth'), strict=False)
        model.head[-1] = nn.Sequential(
            nn.Linear(512, 2, True)
        )
        # model = resnet18(pretrained=True)
        # model.fc = nn.Sequential(
        #     nn.Linear(2048, 2)
        #     # nn.Softmax(dim=1)
        # )
        # print(model)
        # model = VIT_16_256()
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.cuda(device=device_ids[0])
        
        criterion = focal_loss()
        optimizer = optim.Adam(model.parameters(), lr=modellr) #Adam优化器
        # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
        cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=30, eta_min=1e-9)
        
        # 训练模型
        train_model(model, train_loader, val_loader, criterion, optimizer, cosine_schedule)
        
        # 在验证集上进行评估
        correct = 0
        total = 0
        print('val ' + '*' * 30)
        model.eval()
        total_labels = total_outputs = One_hot_label = total_scores = []
        with torch.no_grad():
            for inputs, label in val_loader:
            # for inputs, label in train_loader:
                inputs, label = inputs.cuda(device=device_ids[0]), label.cuda(device=device_ids[0])
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                total_outputs = np.append(total_outputs, outputs.argmax(-1).cpu().numpy())
                total_labels = np.append(total_labels, label.cpu().numpy())
                One_hot_label = np.append(One_hot_label, [int(x) for x in label])
                for row in F.softmax(outputs, dim=1).detach().cpu():
                    total_scores.append(row.tolist())
                 # print(outputs.shape)
                # scores = np.append(scores, outputs.max(1)[0].cpu().numpy())
        One_hot_labels= F.one_hot(torch.from_numpy(np.array(One_hot_label).astype(int))).cpu().tolist()
        accuracy = correct / total
        fold_accuracies.append(accuracy)
        print(type(One_hot_labels[0]))
        print(len(One_hot_labels))
        print(type(total_scores[0]))
        print(len(total_scores))
        print(len(total_scores[0]))
        TestScores.append(total_scores)
        TestLabels.append(One_hot_labels)
        TN, FP, FN, TP = confusion_matrix(total_labels,  total_outputs, labels=[0, 1]).ravel()
        Precision = TP / (TP + FP)
        Recall = TP / (TP+FN)
        Specificity = TN / (TN + FP)
        Sensitivity = TP / (TP + FN)
        F1 =  2 * (Precision * Recall) /  (Precision + Recall)
        print('Acc: {:.4f}'.format(accuracy))
        logger.info('Acc: {:.4f}'.format(accuracy))
        print('Precision: {}'.format(Precision) )
        logger.info('Precision: {}'.format(Precision) )
        print('Recall:    {}'.format(Recall))
        logger.info('Recall:    {}'.format(Recall))
        print('Specificity: {}'.format(Specificity))
        logger.info('Specificity:  {}'.format(Specificity))
        print('Sensitivity: {}'.format(Sensitivity))
        logger.info('Sensitivity:  {}'.format(Sensitivity))
        print('F1_score:  {}'.format(F1))
        logger.info('F1_score:  {}'.format(F1))
        print(f"Fold {fold+1} Accuracy: {accuracy}")
        plotTheCurve(fold+1)    
        print('------------confusion_matrix------------- ')
    #     print(total_outputs.shape)
        print_confusion_matrix(total_labels , total_outputs, 'final_val_', 0)
        # print_roc_curve_final('Val_',fold+1, Labels, Scores)

        print('test ' + '*' * 30)
        correct = 0
        total = 0
        model.eval()
        total_labels = total_outputs = One_hot_label = total_scores = []
        with torch.no_grad():
            for inputs, label in test_loader:
            # for inputs, label in train_loader:
                inputs, label = inputs.cuda(device=device_ids[0]), label.cuda(device=device_ids[0])
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                total_outputs = np.append(total_outputs, outputs.argmax(-1).cpu().numpy())
                total_labels = np.append(total_labels, label.cpu().numpy())
                 # print(outputs.shape)
                One_hot_label = np.append(One_hot_label, [int(x) for x in label])
                for row in F.softmax(outputs, dim=1).detach().cpu():
                    total_scores.append(row.tolist())
        One_hot_labels= F.one_hot(torch.from_numpy(np.array(One_hot_label).astype(int))).cpu().tolist()
        accuracy = correct / total
        test_fold_accuracies.append(accuracy)
        print(type(One_hot_labels[0]))
        print(len(One_hot_labels))
        print(type(total_scores[0]))
        print(len(total_scores))
        print(len(total_scores[0]))
        if fold == num_folds-1: 
            TestScores.append(total_scores)
            TestLabels.append(One_hot_labels)
        TN, FP, FN, TP = confusion_matrix(total_labels,  total_outputs, labels=[0, 1]).ravel()
        Precision = TP / (TP + FP)
        Recall = TP / (TP+FN)
        Specificity = TN / (TN + FP)
        Sensitivity = TP / (TP + FN)
        F1 =  2 * (Precision * Recall) /  (Precision + Recall)
        print('Acc: {:.4f}'.format(accuracy))
        logger.info('Acc: {:.4f}'.format(accuracy))
        print('Precision: {}'.format(Precision) )
        logger.info('Precision: {}'.format(Precision) )
        print('Recall:    {}'.format(Recall))
        logger.info('Recall:    {}'.format(Recall))
        print('Specificity: {}'.format(Specificity))
        logger.info('Specificity:  {}'.format(Specificity))
        print('Sensitivity: {}'.format(Sensitivity))
        logger.info('Sensitivity:  {}'.format(Sensitivity))
        print('F1_score:  {}'.format(F1))
        logger.info('F1_score:  {}'.format(F1))
        print(f"Fold {fold+1} Accuracy: {accuracy}")
        plotTheCurve(fold+1)
        print_roc_curve_final('Test_',fold+1,TestLabels,TestScores)    
    mean_accuracy = sum(fold_accuracies) / num_folds
    test_mean_accuracy = sum(test_fold_accuracies) / num_folds
    print(f"Mean 5-Fold Cross Validation Accuracy: {mean_accuracy}")
    print(f"Mean 5-Fold Cross Validation Test Accuracy: {test_mean_accuracy}")


# model = NET()
# state_dict = torch.load('vit/save_model/100.pth',map_location=torch.device('cpu'))
# # 移除"module."前缀
# new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# model.load_state_dict(new_state_dict, strict=True)
# model.eval()
# # model = model.cuda(device=device_ids[0])
# # criterion = nn.CrossEntropyLoss() #交叉熵函数
# criterion = focal_loss()
k_fold_cross_validation(image_paths, folder_indices, num_folds=5)

# Test(0, model, test_loader, criterion)

# model = NET()
# # criterion = nn.CrossEntropyLoss() #交叉熵函数
# criterion = focal_loss()
# Test(0, model, test_loader, criterion)

# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# for fold, (train_indices, val_indices) in enumerate(kf.split(image_paths, folder_indices)):
#     print(train_indices, val_indices)
#     print(len(train_indices), len(val_indices))
#     traindata = [(image_paths[i],folder_indices[i]) for i in train_indices]

# print('-'*50)
# print('-'*50)
# print('-'*50)
# logger.info('-'*50)
# logger.info('-'*50)
# logger.info('-'*50)
# test_set = MyDataset("/disk/user/lyf/vit/test", transform=train_transforms)
# test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
# Test(1,test_data_loader)



