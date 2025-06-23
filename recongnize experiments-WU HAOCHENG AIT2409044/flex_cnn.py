import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import torch.nn.functional as F
import math
import random

def save_eval_metrics(model, dataloader, class_names,number_model, save_dir=r"C:\Users\吴\Desktop\code\python\principle"):
    model.eval()
    y_true, y_pred = [], []
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    # 保存分类报告
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(2)
    report_path = os.path.join(save_dir, f"classification_report_{number_model}.csv")
    report_df.to_csv(report_path)
    #print(f"✅ 分类报告已保存到 {report_path}")
    # 保存混淆矩阵图
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ")
    cm_path = os.path.join(save_dir, f"confusion_matrix_{number_model}.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    #print(f"✅ 混淆矩阵图已保存到 {cm_path}")
    weighted_precision=report_df.loc["weighted avg"]["precision"]
    return weighted_precision

class FlexibleCNN(nn.Module):
    def __init__(self, num_classes=37, layers_enabled=None,
                 kernel_sizes=None):
        super(FlexibleCNN, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 5, 3, 3, 3, 3, 3]
        if layers_enabled is None:
            layers_enabled = [1, 1, 1, 1, 1, 1, 1, 1]
        self.layers_enabled = layers_enabled
        self.kernel_sizes = kernel_sizes
        self.channel_plan = [16, 32, 64, 128, 128, 256, 256, 512]  # 每层输出通道
        self.convs = nn.ModuleList()
        in_channels = 1  # 初始通道数为1
        for i in range(len(self.channel_plan)):
            if self.layers_enabled[i]:
                conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.channel_plan[i],
                    kernel_size=self.kernel_sizes[i],
                    padding=self.kernel_sizes[i]//2
                )
                self.convs.append(conv)
                in_channels = self.channel_plan[i]  # 更新给下一层
            else:
                self.convs.append(None)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))#自动适配最后尺寸
        self.fc1 = nn.Linear(in_channels, 128)  #使用最后一层输出通道数
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x
        for i, conv in enumerate(self.convs):
            if conv is not None:
                out = F.relu(conv(out))
                if i % 2 == 1:
                    out = self.pool(out)

        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# ---- 训练函数 ----
def train_model(model,num_epochs, train_loader, loss_fn, optimizer,number):
    max_acc=0.0
    patience=10
    count=0
    print("train on the",device)
    for epoch in range(num_epochs):
        start_time=time.time()
        model.train()
        running_loss = 0.0
        correct = 0

        for inputs, labels in train_loader:
            inputs,labels = inputs.to(device),labels.to(device)
            outputs=model(inputs)
            loss=loss_fn(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()   #item是一个数字化的函数
            right_or_wrong=(outputs.argmax(1)==labels)#这个相当于拿出来输出结果列表中的每一个元素和label比较，一样就是1
            correct+=right_or_wrong.sum().item()

        acc = correct/len(train_data)
        print(f"[Epoch {epoch+1}] Train Loss: {running_loss:.4f}  Acc: {acc:.4f}")
        print(f"spend {time.time()-start_time:.2f} s ")
        val_acc=validate(model)
        if max_acc<val_acc:
            torch.save(model, f"cnn_best_{number}.pth")
            print("best model saved")
            count=0
            max_acc=val_acc
        else:
            count+=1
            if count>=patience:
                print("触发早停")
                break
        if val_acc>=0.95:
            print("因为精度已经足够早停")
            break

# ---- 验证函数 ----
def validate(model):
    model.eval()
    correct=0
    with torch.no_grad():
        for inputs,labels in val_loader:
            inputs,labels=inputs.to(device),labels.to(device)
            outputs = model(inputs)
            right_or_wrong=outputs.argmax(1) == labels
            correct += right_or_wrong.sum().item()
    acc = correct / len(val_data)
    print(f"→ Validation Acc: {acc:.4f}")
    return acc


def initial_state():
    layers = [random.choice([0, 1]) for _ in range(8)]
    kernels = [random.choice([3, 5, 7]) for _ in range(8)]
    return layers + kernels


def evaluate_state(state, num_classes, train_loader, val_loader,model_num):
    layers = state[:8]
    kernels = state[8:]
    model = FlexibleCNN(num_classes=num_classes, layers_enabled=layers, kernel_sizes=kernels).to(device)
    # 训练设置
    print(layers+kernels)
    all_labels=[label for _, label in full_dataset.samples]
    weights=compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=all_labels)  # 缓解数据不平衡的问题
    class_weights=torch.tensor(weights, dtype=torch.float).to(device)
    loss_fn=nn.CrossEntropyLoss(weight=class_weights)
    optimizer=optim.Adam(model.parameters(), lr=0.0001)
    train_model(model,150,train_loader,loss_fn,optimizer,model_num)
    result=save_eval_metrics(model, val_loader, number_model=model_num, class_names=full_dataset.classes)
    return result

def perturb(state):
    new_state = state
    while 1:
        idx = random.randint(0,15)
        if idx < 8:
            new_state[idx]=1-new_state[idx]
            if new_state[:8]==[0,0,0,0,0,0,0,0] or new_state[:8]==[1,1,1,1,1,1,1,1]:
                new_state = state
                continue
            break
        else:
            new=random.choice([3, 5, 7])
            if new==new_state[idx]:
                continue
            new_state[idx]=new
            new_state=new_state[:8]+sorted(new_state[8:],reverse=True)
            if new_state==state:
                continue
            break
    return new_state


def simulated_annealing(init_temp, cooling_rate, max_iter):
    min_temp = 1e-3
    time_limit=3*60*60
    start_time = time.time()
    patience = 5
    no_improve_rounds=0
    num_model=11
    best_number=0
    current_state=initial_state()
    best_state = current_state
    best_score = evaluate_state(best_state, num_classes, train_loader, val_loader,num_model)
    temp = init_temp
    for i in range(max_iter):
        new_state = perturb(current_state)
        new_score = evaluate_state(new_state, num_classes, train_loader, val_loader,num_model)
        num_model+=1
        delta = new_score - best_score
        if delta > 0 or random.uniform(0, 1) < math.exp(delta / temp):
            current_state = new_state
            if new_score>best_score:
                no_improve_rounds=0
                best_score=new_score
                best_state=new_state
                best_number=num_model
            else:
                no_improve_rounds+=1
        temp *= cooling_rate
        print(f"迭代 {i + 1}: 当前精度 {new_score:.2f}, 最佳精度 {best_score:.2f}")
        with open(f"C:\\Users\\吴\\Desktop\\code\\python\\principle\\step_result_{num_model}.txt", "w", encoding="utf-8") as f:
            f.write(f"模型编号: {num_model}\n")
            f.write(f"精度: {best_score:.2f}\n")
            f.write(f"状态向量: {best_state}\n")
            f.write(f"迭代数量: {num_model}\n")
            f.write(f"当前温度: {temp}\n")
        if temp < min_temp or no_improve_rounds >= patience or time.time() - start_time > time_limit:
            print("满足终止条件，停止模拟退火")
            break
    return best_state, best_score,best_number,num_model,temp


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Grayscale(),                 # 转为灰度图
    transforms.Resize((64, 64)),            # 调整大小
    transforms.ToTensor(),                  # 转为Tensor
    transforms.Normalize((0.5,), (0.5,))    # 标准化
])



#加载数据集
data_dir =r'C:\Users\吴\Desktop\code\python\principle\bmp'
full_dataset=datasets.ImageFolder(root=data_dir, transform=transform)
#初始化模型
num_classes = len(full_dataset.classes)
#训练设置
all_labels = [label for _, label in full_dataset.samples]
#划分训练,验证集
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=True)
#训练模型

best_state, best_score,best_number,num_model,temp=simulated_annealing(0.09414317882700003,0.9,20)
with open(r"C:\Users\吴\Desktop\code\python\principle\best_result.txt", "w", encoding="utf-8") as f:
    f.write(f"最佳模型编号: {best_number}\n")
    f.write(f"最佳精度: {best_score:.2f}\n")
    f.write(f"最佳状态向量: {best_state}\n")
    f.write(f"迭代数量: {num_model}\n")
    f.write(f"当前温度: {temp}\n")