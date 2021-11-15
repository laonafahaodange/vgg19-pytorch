import os
import json
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from vgg import VGG

BATCH_SIZE = 256  # 论文256
LR = 0.01  # 论文 0.01
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
EPOCHS = 74  # 论文74

DATASET_PATH = 'data'
MODEL = 'vgg19.pth'


def train_device(device='cpu'):
    # 只考虑单卡训练
    if device == 'gpu':
        cuda_num = torch.cuda.device_count()
        if cuda_num >= 1:
            print('device:gpu')
            return torch.device(f'cuda:{0}')
    else:
        print('device:cpu')
        return torch.device('cpu')


def dataset_loader(dataset_path):
    dataset_path = os.path.join(os.getcwd(), dataset_path)
    assert os.path.exists(dataset_path), f'[{dataset_path}] does not exist.'
    train_dataset_path = os.path.join(dataset_path, 'train')
    val_dataset_path = os.path.join(dataset_path, 'val')

    # 和alexnet差不多
    data_transform = {
        'train': transforms.Compose([transforms.RandomResizedCrop(size=224),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'val': transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(root=val_dataset_path, transform=data_transform['val'])
    return train_dataset, val_dataset


def idx2class_json(train_dataset):
    class2idx_dic = train_dataset.class_to_idx
    idx2class_dic = dict((val, key) for key, val in class2idx_dic.items())
    # json.dumps()把python对象转换成json格式的字符串
    json_str = json.dumps(idx2class_dic)
    with open('class_idx.json', 'w') as json_file:
        json_file.write(json_str)
    print('write class_idx.json complete.')


def evaluate_val_accuracy(net, val_dataset_loader, val_dataset_num, device=torch.device('cpu')):
    # ==============================================
    # isinstance()与type()区别：
    # type()不会认为子类是一种父类类型，不考虑继承关系。
    # isinstance()会认为子类是一种父类类型，考虑继承关系。
    # 如果要判断两个类型是否相同推荐使用isinstance()
    # ==============================================
    if isinstance(net, nn.Module):
        net.eval()
    val_correct_num = 0
    for i, (val_img, val_label) in enumerate(val_dataset_loader):
        val_img, val_label = val_img.to(device), val_label.to(device)
        output = net(val_img)
        _, idx = torch.max(output.data, dim=1)
        val_correct_num += torch.sum(idx == val_label)
    val_correct_rate = val_correct_num / val_dataset_num
    return val_correct_rate


def train(net, train_dataset, val_dataset, device=torch.device('cpu')):
    train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
    print(f'[{len(train_dataset)}] images for training, [{len(val_dataset)}] images for validation.')
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)  # 论文使用的优化器
    # optimizer = optim.Adam(net.parameters(), lr=0.0002)
    # 学习率调整策略
    # vgg和alexnet的策略一样，都是将错误率（应该指的是验证集）作为指标，当错误率不再下降的时候降低学习率。vgg训练了大约74个epoch，学习率下降3次
    # 第一种策略，每24个epoch降低一次学习率（不严谨）
    # lr_scheduler=optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30,gamma=0.1)
    # 第二种策略，错误率不再下降的时候降低学习率，我们后面会计算验证集的准确率，错误率不再下降和准确率不再提高是一个意思,所以mode为max
    # ==================================================================================================================
    # 注意：在PyTorch 1.1.0之前的版本，学习率的调整应该被放在optimizer更新之前的。如果我们在 1.1.0 及之后的版本仍然将学习率的调整
    # （即 scheduler.step()）放在 optimizer’s update（即 optimizer.step()）之前，那么 learning rate schedule 的第一个值将
    # 会被跳过。所以如果某个代码是在 1.1.0 之前的版本下开发，但现在移植到 1.1.0及之后的版本运行，发现效果变差，
    # 需要检查一下是否将scheduler.step()放在了optimizer.step()之前。
    # ==================================================================================================================
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.1, patience=24,
                                                        min_lr=0.00001)
    # 在训练的过程中会根据验证集的最佳准确率保存模型
    best_val_correct_rate = 0.0
    for epoch in range(EPOCHS):
        net.train()
        # 可视化训练进度条
        train_bar = tqdm(train_dataset_loader)
        # 计算每个epoch的loss总和
        loss_sum = 0.0
        for i, (train_img, train_label) in enumerate(train_bar):
            optimizer.zero_grad()
            train_img, train_label = train_img.to(device), train_label.to(device)
            output = net(train_img)
            loss = loss_function(output, train_label)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            train_bar.desc = f'train epoch:[{epoch + 1}/{EPOCHS}], loss:{loss:.5f}'
        # 测试验证集准确率
        val_correct_rate = evaluate_val_accuracy(net, val_dataset_loader, len(val_dataset), device)
        # 根据验证集准确率更新学习率
        lr_scheduler.step(val_correct_rate)
        print(
            f'epoch:{epoch + 1}, '
            f'train loss:{(loss_sum / len(train_dataset_loader)):.5f}, '
            f'val correct rate:{val_correct_rate:.5f}')
        if val_correct_rate > best_val_correct_rate:
            best_val_correct_rate = val_correct_rate
            # 保存模型
            torch.save(net.state_dict(), MODEL)
    print('train finished.')


if __name__ == '__main__':
    # 这里数据集只有5类
    vgg19 = VGG(num_labels=5)
    # 加载vgg19的预训练模型
    pretrain_model = torch.load('vgg19-dcbb9e9d.pth')
    # pretrain_model是以字典的形式读取出预训练模型的权重参数，vgg19_dict表示自定义模型的参数字典
    # state_dict存储自定义模型和预训练模型的共有参数
    vgg19_dict = vgg19.state_dict()
    state_dict = {k: v for k, v in pretrain_model.items() if k in vgg19_dict.keys()}
    # 由于预训练模型最后一层全连接层的输出为1000类，我们自定义的输出为5类，所以要去掉最后一层全连接层的权重以及bias（最后一层全连接层用随机初始化）
    state_dict.pop('classifier.6.weight')
    state_dict.pop('classifier.6.bias')
    # vgg19_dict对共有参数进行更新
    vgg19_dict.update(state_dict)
    # 自定义模型加载参数
    vgg19.load_state_dict(vgg19_dict)
    device = train_device('gpu')
    train_dataset, val_dataset = dataset_loader(DATASET_PATH)
    # 保存类别对应索引的json文件，预测用
    idx2class_json(train_dataset)
    train(vgg19, train_dataset, val_dataset, device)
