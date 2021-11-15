import os
from shutil import copy, rmtree
import random


def make_dir(file_path):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹再创建
        rmtree(file_path)
    os.makedirs(file_path)


def split_data(input_file_path, output_file_path, split_rate, seed='random'):
    if seed == 'fixed':
        random.seed(0)
    else:
        random.seed()
    # 获取当前文件路径
    cwd = os.getcwd()
    input_dataset_path = os.path.join(cwd, input_file_path)
    output_dataset_path = os.path.join(cwd, output_file_path)
    assert os.path.exists(input_dataset_path), f"path '{input_dataset_path}' does not exist."
    # ===================#######################################
    # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    # os.path.isdir() 方法用于判断某一路径是否为目录
    # 先遍历dataset_path获得文件\文件夹名称列表，再判断名称是否为目录
    ############################################################
    dataset_classes = [dataset_class for dataset_class in os.listdir(input_dataset_path) if
                       os.path.isdir(os.path.join(input_dataset_path, dataset_class))]
    # 训练集
    train_path = os.path.join(output_dataset_path, 'train')
    make_dir(train_path)
    for dataset_class in dataset_classes:
        make_dir(os.path.join(train_path, dataset_class))
    # 验证集
    val_path = os.path.join(output_dataset_path, 'val')
    make_dir(val_path)
    for dataset_class in dataset_classes:
        make_dir(os.path.join(val_path, dataset_class))

    for dataset_class in dataset_classes:
        input_dataset_class_path = os.path.join(input_dataset_path, dataset_class)
        images = os.listdir(input_dataset_class_path)
        images_num = len(images)
        # 随机选取验证集
        val_images = random.sample(images, k=int(images_num * split_rate))
        for index, image in enumerate(images):
            # 获取图像路径
            image_path = os.path.join(input_dataset_class_path, image)
            if image in val_images:
                # 将图像文件copy到验证集对应路径
                copy(image_path, os.path.join(val_path, dataset_class))
            else:
                copy(image_path, os.path.join(train_path, dataset_class))
            print(f'[{dataset_class}] is processing: {index + 1}/{images_num}')
    print('process finished.')


'''
文件结构示例：
|-- split_data.py
|-- flower_photos
    |-- daisy
    |-- dandelion
    |-- roses
    |-- sunflowers
    |-- tulips
    |-- LICENSE.txt
|-- data
    |-- train
        |-- daisy
        |-- dandelion
        |-- roses
        |-- sunflowers
        |-- tulips
    |-- val
        |-- daisy
        |-- dandelion
        |-- roses
        |-- sunflowers
        |-- tulips    
'''
if __name__ == '__main__':
    original_data_file_path = 'flower_photos'
    spilit_data_file_path = 'data'
    split_rate = 0.1
    split_data(original_data_file_path, spilit_data_file_path, split_rate)
