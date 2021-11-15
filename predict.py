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
