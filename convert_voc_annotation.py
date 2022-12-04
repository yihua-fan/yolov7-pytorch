import os
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm
from utils.utils import get_classes

classes_path = 'model_data/voc_classes.txt'
classes, _ = get_classes(classes_path)


# get class and location coordinates from xml file
def convert_annotation(in_file_path, out_file, class_nums):
    in_file = open(in_file_path)
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        class_nums[cls_id] += 1
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        out_file.write(' ' + ','.join([str(a) for a in b]) + ',' + str(cls_id))


# generate test.txt for testing
def generate_test_txt(annonation_dir, txt_dir):
    out_file_path = os.path.join(txt_dir, 'test.txt')
    out_file = open(out_file_path, 'w')
    class_nums = [0 for cls in classes]
    total_num = len(os.listdir(annonation_dir))
    with tqdm(total=total_num, desc='Convert') as pbar:
        for file_name in os.listdir(annonation_dir):
            in_file_path = os.path.join(annonation_dir, file_name)
            img_path = in_file_path.replace('Annotations', 'JPEGImages').replace('xml', 'jpg')
            img_path = os.path.abspath(img_path)
            out_file.write(img_path)
            convert_annotation(in_file_path, out_file, class_nums)
            out_file.write('\n')
            pbar.update(1)
    out_file.close()

    print(f'=> Test Data: {total_num}')
    for i in range(len(classes)):
        print(f'{classes[i]}: {class_nums[i]}', end='   ')


# generate train.txt and val.txt for training (random split train set)
def generate_train_val_txt(annonation_dir, txt_dir, train_ratio):
    train_file_path = os.path.join(txt_dir, 'train.txt')
    val_file_path = os.path.join(txt_dir, 'val.txt')
    train_file = open(train_file_path, 'w')
    val_file = open(val_file_path, 'w')
    train_class_nums = [0 for cls in classes]
    val_class_nums = [0 for cls in classes]

    total_num = len(os.listdir(annonation_dir))
    train_num = int(total_num * train_ratio)
    val_num = total_num - train_num
    val_set = random.sample(os.listdir(annonation_dir), val_num)

    with tqdm(total=total_num, desc='Convert') as pbar:
        for file_name in os.listdir(annonation_dir):
            in_file_path = os.path.join(annonation_dir, file_name)
            img_path = in_file_path.replace('Annotations', 'JPEGImages').replace('xml', 'jpg')
            img_path = os.path.abspath(img_path)
            if file_name in val_set:
                val_file.write(img_path)
                convert_annotation(in_file_path, val_file, val_class_nums)
                val_file.write('\n')
            else:
                train_file.write(img_path)
                convert_annotation(in_file_path, train_file, train_class_nums)
                train_file.write('\n')
            pbar.update(1)
    train_file.close()
    val_file.close()

    print(f'=> Train Data: {train_num}')
    for i in range(len(classes)):
        print(f'{classes[i]}: {train_class_nums[i]}', end='   ')

    print(f'\n=> Val Data: {val_num}')
    for i in range(len(classes)):
        print(f'{classes[i]}: {val_class_nums[i]}', end='   ')


if __name__ == '__main__':
    txt_dir = 'data'
    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)

    annonation_dir = r'C:\Users\fyh\Desktop\voc-fog(9578+2129)\train\Annotations'
    train_ratio = 0.9
    generate_train_val_txt(annonation_dir, txt_dir, train_ratio)

    annonation_dir = r'C:\Users\fyh\Desktop\voc-fog(9578+2129)\test\Annotations'
    generate_test_txt(annonation_dir, txt_dir)
