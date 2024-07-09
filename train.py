import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import csv
# 加载自定义模型和数据集
# import resnet02
# import resnet_original
# import U_net
# import Vggnet
import efficientnet
# import cspdarknet
# import VIT
# import squeezenet
# import AlexNet2_1
# import RMT
from pokemon import Pokemon
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,auc
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import math
from torch.nn.parallel import DataParallel
# 创建自定义数据集实例
# train_dataset = MyDataset(train=True)
# test_dataset = MyDataset(train=False)
train_dataset = Pokemon('pokemon', mode='train')
# x,y = next(iter(train_db))
test_dataset = Pokemon('pokemon', mode='val')

# 处理样本不平衡 a:tensor([4., 9.]) NoPD和PD的数量  b:NoPD和PD的集合
a, b = train_dataset.get_class_counts()
# print("a:", a)
class_counts = a
class_weights = 1.0 / class_counts.float()  # NoPD的权重为1/4 PD的权重为1/9
# print("class_weights:",class_weights) #tensor([1/4, 1/9])
sample_weights = class_weights[b]  # 将样本的权重，并将其保存在sample_weights张量中。
# print("sample_weights:",sample_weights) #tensor([1/4, 1/9, 1/9, 1/9, 1/4, .....])
sampler = torch.utils.data.WeightedRandomSampler(sample_weights,
                                                 len(sample_weights))  # 使用WeightedRandomSampler创建一个采样器sampler，并将其传递给训练数据加载器

# 创建数据加载器
batch_size =1
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化自定义模型和优化器
# model = resnet_original.ResNet18()
# model = AlexNet2_1.AlexNet()
# model = VIT.ViT3D()
# model = U_net.UNet3D()
# model = Vggnet.VGGNet3D(2)
model = efficientnet.EfficientNetB0(1,2)
# model = cspdarknet.CSPDarknet()
# model = RMT.VisRetNet()
# 预训练
# checkpoint = torch.load('best_Unet.pth')
# # print("------------------",checkpoint.keys())
# pretrained_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
# model.load_state_dict(pretrained_state_dict)
model = nn.DataParallel(model)
# model = squeezenet.SqueezeNet()


init_lr = 0.006
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005,weight_decay=0,amsgrad=True)
optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, nesterov=True, weight_decay=0.001)


# lr = torch.optim.lr_scheduler.StepLR(optimizer,step_size=15,gamma=0.5)
def set_cosine_lr(optimizer, current_epoch, max_epoch, lr_min=0., lr_max=0.1, warmup=True, num_warmup=5):

    warmup_epoch = num_warmup if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    elif current_epoch < max_epoch:
        lr = lr_min + (lr_max - lr_min) * (
                1 + math.cos(math.pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    else:
        lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * (current_epoch - max_epoch) / max_epoch)) / 2
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
# 画图
train_accuracy_history = []
train_loss_history = []
test_accuracy_history = []
test_loss_history = []
precision_history = []
recall_history = []
f1_history = []
auc1_history = []



# 训练过程
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda')
model = model.to(device)

# model = model.to('cuda')
# print("开始epoch")
resnet_original_file = 'resnet_original.csv'
with open(resnet_original_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        ['Epoch', 'Train_accuracy', 'Train_loss', 'Test_accuracy', 'Test_loss', 'Precision', 'Recall', 'F1', 'AUC','roc_fpr','roc_tpr','specificity'])
roc_file = 'roc.csv'
with open(roc_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'fpr', 'tpr', 'thresholds'])

best_test_acc = 0
# 判断auc1,防止预测值为同一类时报错
def calculate_auc1(test_targets, test_predictions):
    if len(set(test_targets)) == 1:
        return 0.001
    else:
        return roc_auc_score(test_targets, test_predictions)

for epoch in range(num_epochs):
    print("epoch:", epoch)
    lr = set_cosine_lr(optimizer, epoch, num_epochs, lr_min=init_lr * 0.01, lr_max=init_lr)
    print("train_lr:", lr)
    train_accuracy = 0
    train_loss_total = 0
    train_loss_avg = 0
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        # print("data:",data)
        # print("train_target:",targets)
        model.train()
        optimizer.zero_grad()
        data = data.to(device)
        targets = targets.to(device)
        train_outputs = model(data)
        train_loss = criterion(train_outputs, targets)
        train_loss_total += train_loss.item()
        # print("train_outputs:",train_outputs)
        train_predictions = torch.argmax(train_outputs, dim=1)
        # print(targets,train_predictions)
        train_accuracy += accuracy_score(targets.cpu(), train_predictions.cpu())
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
    train_accuracy = train_accuracy / len(train_loader)
    train_loss_avg = train_loss_total / len(train_loader)

    # lr.step()

    # 模型评估
    test_accuracy, precision, recall, f1, auc1 = 0, 0, 0, 0, 0,
    tn, fp, fn, tp=0,0,0,0
    test_loss_total = 0
    test_loss_avg = 0
    all_predictions = []
    all_targets = []
    all_scores=[]
    with torch.no_grad():
        for batch_idx, (data, test_targets) in enumerate(test_loader):
            model.eval()
            data = data.to(device)
            test_targets = test_targets.to(device)
            test_outputs = model(data)
            test_predictions = torch.argmax(test_outputs, dim=1)
            print(test_targets, test_predictions)
            test_accuracy += accuracy_score(test_targets.cpu(), test_predictions.cpu())
            test_loss = criterion(test_outputs, test_targets)
            test_loss_total += test_loss.item()
            precision += precision_score(test_targets.cpu(), test_predictions.cpu(), zero_division=1)
            recall += recall_score(test_targets.cpu(), test_predictions.cpu())
            f1 += f1_score(test_targets.cpu(), test_predictions.cpu())
            all_predictions.extend(test_predictions.cpu().tolist())
            all_targets.extend(test_targets.cpu().tolist())
            all_scores.extend(test_outputs[:, 1].tolist())
            # auc1 += roc_auc1_score(test_targets.cpu(), test_predictions.cpu())
            auc1 += calculate_auc1(test_targets=test_targets.cpu(), test_predictions=test_predictions.cpu())
            # print("precision,recall,f1,auc1",precision,recall,f1,auc1)

    test_accuracy = test_accuracy / len(test_loader)
    precision = precision / len(test_loader)
    recall = recall / len(test_loader)
    f1 = f1 / len(test_loader)
    auc1 = auc1 / len(test_loader)
    test_loss_avg = test_loss_total / len(test_loader)
    tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()
    specificity = tn / (tn + fp)

    if test_accuracy > best_test_acc:
        torch.save(model.state_dict(), 'best_Unet.pth')
        fpr, tpr, thresholds = roc_curve(all_targets, all_scores)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (area = %0.2f)' % auc1)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('pokemon_MRI1/image_roc.png', dpi=300, bbox_inches='tight')
        with open(roc_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, fpr,tpr,thresholds])
    with open(resnet_original_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, train_accuracy, train_loss_avg, test_accuracy, test_loss_avg,
                         precision, recall, f1, auc1,specificity])


    if epoch % 50 == 0:  # 每隔50轮打印一次梯度信息
        print(f"Gradient info at epoch {epoch}:")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Layer: {name}, Gradient Norm: {torch.norm(param.grad)}")

    # 保存指标值到相应的列表中
    train_accuracy_history.append(train_accuracy)
    train_loss_history.append(train_loss_avg)
    test_accuracy_history.append(test_accuracy)
    test_loss_history.append(test_loss_avg)
    precision_history.append(precision)
    recall_history.append(recall)
    f1_history.append(f1)
    auc1_history.append(auc1)



    # 打印当前epoch的指标
    print(
        f"Epoch [{epoch + 1}/{num_epochs}] - Train Acc: {train_accuracy:.4f}, Train Loss: {train_loss_avg:.4f}, Test Acc: {test_accuracy:.4f}, Test Loss: {test_loss_avg:.4f}, "
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f},AUC: {auc1:.4f},roc_fpr:{fpr:.4f},roc_tpr:{tpr:.4f},specificity:{specificity:.4f}")


plt.figure()
plt.plot(train_accuracy_history, label='Train Accuracy')
plt.plot(test_accuracy_history, label='Test Accuracy')
plt.xlabel('Epoch')
plt.legend()
# plt.show()
plt.savefig('pokemon_MRI1/image_acc.png', dpi=300, bbox_inches='tight')

# 绘制预测准确率和损失曲线


# 绘制精确度、召回率、F1和AUC曲线
plt.figure()
plt.plot(precision_history, label='Precision')
plt.plot(recall_history, label='Recall')
plt.plot(f1_history, label='F1')
plt.plot(auc1_history, label='AUC')
plt.xlabel('Epoch')
plt.legend()
# plt.show()
plt.savefig('pokemon_MRI1/image_prfa.png', dpi=300, bbox_inches='tight')



plt.figure()
plt.plot(train_loss_history, label='Train Loss')
plt.plot(test_loss_history, label='Test Loss')
plt.xlabel('Epoch')
plt.legend()
# plt.show()
plt.savefig('pokemon_MRI1/image_loss.png', dpi=300, bbox_inches='tight')

    # 绘制roc曲线
# plt.figure()
# plt.plot(fpr_history, tpr_history, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc1_history[-1])
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.savefig('pokemon_MRI1/image_roc.png', dpi=300, bbox_inches='tight')