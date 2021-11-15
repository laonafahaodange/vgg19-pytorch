# VGG
VGG是牛津大学的视觉几何组（Visual Geometry Group）在2015年的论文《Very Deep Convolutional Networks for Large-Scale Image Recognition》上提出的一种结构，在当年（2014年）的ImageNet分类挑战取得了第二名的好成绩（第一名是GoogleNet）。主要工作是证明了通过使用非常小的卷积层（3x3）来**增加网络深度进而提高性能**。

## VGG结构
论文共设计了5种网络（A~E）来证明网络深度的增加带来的网络性能的提升。
![](images/VGG.png 'VGG')
激活函数采用relu，除了A-LRN以外都没有使用LRN，而且发现LRN并不能提高性能，还增加了计算量和计算开销。  
尽管深度增加了，但是参数并不多于使用大的卷积层的浅层网络（例如Sermanet等人所用的网络，有144M参数）。
![](images/VGG_params.png 'VGG参数')
整个VGG网络使用了非常小的感受野（3x3），步长为1。论文提到2个3x3的卷积层等效于1个5x5的卷积层；3个3x3的卷积层等效于1个7x7的卷积层。  
感受野的定义如下：  

> 感受野（Receptive Field），指的是神经网络种神经元“看到的”输入区域，在卷积神经网络中，feature map上某个元素的计算受输入图像上某个区域的影响，这个区域即该元素的感受野。

根据定义，我们假设一个7x7的卷积层，则感受野为7x7的一个范围，如图所示
![](images/example1.png '7x7感受野示意图')

根据公式
$$ M=\lfloor\frac{N-kernelsize+2*padding}{stride}+1\rfloor$$

- N为输入尺寸，M为输出尺寸

代入padding=0，stride=1，我们可以依次计算出经过3个3x3卷积层后输出的尺寸分别为5、3、1，如图所示
![](images/example.png '经过3个3x3的卷积层感受野示意图')
可以看到，1个7x7卷积层的感受野等效于经过3个3x3的卷积层的感受野；论文提到的经过2个3x3的卷积层等效于1个5x5的卷积层同理。  

采用连续3个的3x3卷积层而不是单个7x7卷积层的原因：  
- 相比于单个7x7卷积层只有一个非线性层（论文里面指的是relu），3个3x3卷积层可以引入3个非线性层（个人理解这样子可以提高特征的表达能力）。
- 相比于单个7x7卷积层，3个3x3的卷积层可以减少参数数量。例如，假设卷积层的输入输出通道都为C，那么3个3x3卷积层的参数数量为$3\times(3^2\times C^2)=27C^2$，而1个7x7卷积层的参数数量为$7^2\times C^2=49C^2$

3个3x3卷积层可以看作是对1个7x7的卷积层施加了正则化（对模型的参数进行处理），通过3个3x3的卷积层分解1个7x7的卷积层。

## 训练
训练图像为224x224的RGB图像，做零均值预处理。  
### 训练策略：
- 使用带momentum的MBGD（mini-batch gradient decent）优化多类逻辑回归
- batchsize=256，momentum=0.9，$L_2$正则惩罚设置为$5\cdot 10^{-4}$，lr=0.01
- 前两层全连接层使用p=0.5的dropout
- 验证集准确率停止增加时学习率除以10，学习率共减少三次
- 在ImageNet数据集上共370K次迭代，74个epoch
- 5个网络的权重初始化略有不同，例如训练浅层网络A采用了随机初始化，而训练更深层的网络时，前4层卷积层以及最后3层全连接层初始化为网络A训练后的权重，而中间层则随机初始化

论文也提到相比于alexnet，VGG的参数数量和深度都有增加，但是网络收敛所需的epoch变少了（alexnet训练了90个epoch），论文作者推测是以下原因：
1. 通过网络深度的增加以及更小尺寸的卷积层隐式地使用了正则化
2. 某些具体层的预初始化，作者也提到了网络权重的初始化非常重要。由于深层网络梯度的不稳定性，不当的初始化可能导致学习停止
### 训练图片的尺寸
作者定义**训练尺度S**为isotropically-rescaled（即宽、高一致的rescale）后图像的最短边，S肯定不能小于crop size（裁剪尺寸，作者定义为224x224）。当S=crop size时，裁剪的图像获得的是整个图像的统计信息。当S远大于224时，裁剪的图像获得的是图像的部分信息。  
作者考虑了两种策略设置S：
1. 固定S为256和384（为了加速S=384时网络的训练，用了S=256时的预训练并且使用了更小的初始学习率$lr=10^{-3}$）
2. 多尺度训练(scale jitter)，即每张训练图片随机从一个固定的范围$[S_{min},S_{max}]$内采样进行rescale（论文$S_{min}=256,S_{max}=512$）。由于同一个目标在不同的图像内的大小可能是不一样的，所以采用多尺度训练是有益的（有点像yolo）
## 测试
与训练尺度类似，定义**测试尺度Q**，对每种训练尺度S采用几种不同的测试尺度Q也会提升性能。作者注意到Q没有必要等于S。网络以类似Sermanet等人的方式密集地应用于rescale的测试集图像，即第一层全连接层转换为7x7的卷积层，后两层全连接层转换为1x1卷积层。然后将全卷积网络应用到整个未裁剪的图像。  
全连接层转换为卷积层的理解：  
VGG结构的卷积层不改变输入的尺寸，但是池化层会改变，论文使用的是stride为2的2x2最大池化。根据公式
$$ M=\lfloor\frac{N-kernelsize+2*padding}{stride}+1\rfloor$$
可以计算出第一层全连接层的输入为$512\times 7 \times 7=25088$，从参数数量的角度来说，全连接层和7x7卷积层的参数数量都是$512\times 4096\times 7\times 7$。但是全连接和卷积层的计算方式还是略有不同的（但是数学本质是一样的，都是特征图中的元素乘以权重再求和）。
> 个人理解全连接层相当于对整个输入的feature map（25088个元素）做4096个feature map size的卷积，每个卷积的结果就是一个数，作为下一层feature map的一个元素；而7x7的卷积层则把输入的feature map划分成512份，每份的size为7x7，同样用$4096\times 512$个但是size为7x7的卷积层做卷积，再沿着512维方向做叠加，得到输出feature map，对于输出feature map中的每一个元素，相当于对整个输入的feature map做了512次7x7的卷积再求和（算不算一种池化？）。

作者也采用了水平翻转图像的数据增强策略，根据原始图像和翻转图像的soft-max类后验进行平均，以获得图像的最终分数。
> 这里顺便引用一下原文的翻译：
> 由于全卷积网络被应用在整个（或者说uncrop，未裁剪的）图像上，所以不需要在测试时采样多个裁剪图像，因为这需要网络重新计算每个裁剪图像，效率较低。同时，使用大量的裁剪图像又可以提高准确度，因为与全卷积网络相比，它对输入图像的采样更精细。此外，由于不同的卷积边界条件，多裁剪图像评估是对密集评估的补充：当将ConvNet应用于裁剪图像时，卷积特征图用零填充，而在密集评估的情况下，相同裁剪图像的填充自然会来自于图像的相邻部分（由于卷积和空间池化），这大大增加了整个网络的感受野，因此捕获了更多的上下文。虽然我们认为在实践中，多裁剪图像的计算时间增加并不足以证明准确性的潜在收益，但作为参考，我们还是在每个尺度使用50个裁剪图像（5×5网格和2次翻转）评估了我们的网络，在3个尺度上总共150个裁剪图像。

## 分类实验
### 数据集
数据集采用ILSVRC-2012，分类性能采用top-1和top-5 error进行评估。
> top-1 = （正确标记 与 模型输出的最佳标记不同的样本数）/ 总样本数；
> top-5 = （正确标记 不在 模型输出的前5个最佳标记中的样本数）/ 总样本数；
> top-1即取输出概率向量里最大值对应的标签（label）作为预测标签，如分类正确，则预测正确。否则预测错误。
> top-5即取输出概率向量前5个最大值对应的标签，只要这5个标签内包含正确的分类标签，则预测正确。否则预测错误。

### 单尺度评估
固定测试集图片的size。另外作者发现LRN并不能提高模型性能，所以网络B-E都没有采用LRN。其次作者发现分类的错误率随着网络深度的增加而减小，这说明尽管额外的非线性确实有帮助，但使用具有非凡（non-trivial）感受野的卷积层来捕获空间上下文也同样重要。测试集单尺度评估的结果如图所示
![](images/singletestscale.png '单尺度测试评估')

### 多尺度评估
测试集图片采取多个size，对类别后验结果取平均。考虑到训练和测试的尺度如果差异太大会导致性能的急剧下降，根据S的形式采用不同的策略
- 如果训练尺度S为固定值，则$Q=\{S-32,S,S+32\}$
- 如果训练尺度S为多尺度，即$S\in [S_{min},S_{max}]$，则$Q=\{S_{min},0.5(S_{min}+S_{max}),S_{max}\}$
多尺度评估结果如图所示
![](images/multitestscale.png '多尺度评估')

### 多裁剪图像评估
> 原文翻译：
> 我们将密集评估（应该是上面提到的全连接层换成卷积层的网络）与多裁剪图像评估进行比较。我们还通过平均其soft-max输出来评估两种评估技术的互补性。可以看出，使用多裁剪图像表现比密集评估略好，而且这两种方法确实是互补的，因为它们的组合优于其中的每一种。如上所述，我们假设这是由于卷积边界条件的不同处理所造成的。

多裁剪图像评估结果
![](images/multicrop.png '多裁剪图像评估')

### 卷积网络融合
作者通过对soft-max类别后验概率进行平均，融合了多种模型的输出。由于模型的互补性，性能得到了提高。
![](images/multiconvnetfusion.png '卷积网络融合')
此外作者发现采用表现最好的D和E进行融合效果也不错（只融合了两个模型，显著少于当时其它参赛组融合的模型数量）。
![](images/comparisonmodelfusion.png '卷积网络融合比较')

## 结论
深度有利于分类精度，增加传统卷积网络结构的深度可以实现更好的性能。证实了深度在视觉表示中的重要性。

---

# 代码部分
数据集下载地址：<http://download.tensorflow.org/example_images/flower_photos.tgz>
vgg19预训练模型：<https://download.pytorch.org/models/vgg19-dcbb9e9d.pth>
代码实现的是VGG19，与之前alexnet的pytorch实现类似，主要包含数据集划分代码`spilit.py`，模型代码`vgg.py`，训练代码`train.py`和预测代码`predict.py`。详细见GitHub，这里只放了部分代码。
## 模型
```
import torch
from torch import nn


# VGG19
class VGG(nn.Module):
    def __init__(self, num_labels=1000):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_labels)
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01 ** 2)
                nn.init.constant_(layer.bias, val=0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01 ** 2)
                nn.init.constant_(layer.bias, val=0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def test_output_shape(self):
        # out_channel input_channel, image_size
        test_img = torch.rand(size=(1, 3, 227, 227), dtype=torch.float32)
        for layer in self.features:
            test_img = layer(test_img)
            print(layer.__class__.__name__, 'output shape: \t', test_img.shape)

# vgg19 = VGG(num_labels=5)
# vgg19.test_output_shape()


```
## 训练
训练的时候使用了pytorch提供的的vgg19预训练模型权重，也是这时候发现之前alexnet在训练时候验证集准确率卡在24%的原因。加载权重的具体方式参考代码注释。
```
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

```

## 预测
```
import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vgg import VGG

IMG_PATH = 'test_img/tulip.jpg'
JSON_PATH = 'class_idx.json'
WEIGHT_PATH = 'vgg19.pth'


def predict(net, img, json_label):
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    original_img=img
    img = data_transform(img)  # 3,224,224
    img = torch.unsqueeze(img, dim=0)  # 1,3,224,224
    assert os.path.exists(WEIGHT_PATH), f'file {WEIGHT_PATH} does not exist.'
    net.load_state_dict(torch.load(WEIGHT_PATH))
    net.eval()
    with torch.no_grad():
        output = torch.squeeze(net(img))  # net(img)的size为1,5，经过squeeze后变为5
        predict = torch.softmax(output, dim=0)
        predict_label_idx=int(torch.argmax(predict))
        predict_label=json_label[str(predict_label_idx)]
        predict_probability=predict[predict_label_idx]
    predict_result=f'class:{predict_label}, probability:{predict_probability:.3f}'
    plt.imshow(original_img)
    plt.title(predict_result)
    print(predict_result)
    plt.show()


def read_json(json_path):
    assert os.path.exists(json_path), f'{json_path} does not exist.'
    with open(json_path, 'r') as json_file:
        idx2class = json.load(json_file)
        return idx2class


if __name__ == '__main__':
    net = VGG(num_labels=5)
    img = Image.open(IMG_PATH)
    idx2class = read_json(JSON_PATH)
    predict(net, img, idx2class)

```

### 预测结果
测试图片是网上随便找的5张图片（不排除训练集已经出现过了），预测分类全部正确。
![](predict_result/daisy.png 'daisy')
![](predict_result/dandelion.png 'dandelion')
![](predict_result/rose.png 'rose')
![](predict_result/sunflower.png 'sunflower')
![](predict_result/tulip.png 'tulip')

## 总结
正如作者提到的那样，网络权重的初始化非常重要。由于深层网络梯度的不稳定性，不当的初始化可能导致学习停止。一开始我在kaggle上训练的时候用的是随机初始化（作者在论文里面对模型A的随机初始化）也发现验证集的准确率和之前在alexnet的训练一样卡在24%，后来尝试加载预训练模型权重后发现改善非常巨大，按照论文的方法训练74个poch（在kaggle上花了90分钟左右），验证集准确率最高达到94%。


--- 
参考：  
- [一文读懂VGG网络]<https://zhuanlan.zhihu.com/p/41423739>
- [彻底搞懂感受野的含义与计算]<https://www.cnblogs.com/shine-lee/p/12069176.html>
- [卷积神经网络VGG 论文细读 + Tensorflow实现]<https://www.jianshu.com/p/68ac04943f9e>
- [卷积神经网络之VGG]<https://zhuanlan.zhihu.com/p/116900199>
- [VGG-Net论文解析]<https://www.cnblogs.com/kk17/p/9792071.html>
- [VGGNet论文翻译]<https://www.cnblogs.com/bigcindy/p/10688835.html>
- [利用Pytorch加载预训练模型的权重]<https://www.cnblogs.com/vvzhang/p/14924222.html>