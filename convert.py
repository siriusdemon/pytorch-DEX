# quick and dirty convertion script
import caffe
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Age, Gender



def convert_age():
    torch_net = Age()

    caffe_net = caffe.Net('age.prototxt', "dex_chalearn_iccv2015.caffemodel", caffe.TEST)
    caffe_params = caffe_net.params

    mappings = {
        'conv1_1': torch_net.conv[0].conv1,
        'conv1_2': torch_net.conv[0].conv2,
        'conv2_1': torch_net.conv[1].conv1,
        'conv2_2': torch_net.conv[1].conv2,
        'conv3_1': torch_net.conv[2].conv1,
        'conv3_2': torch_net.conv[2].conv2,
        'conv3_3': torch_net.conv[2].conv3,
        'conv4_1': torch_net.conv[3].conv1,
        'conv4_2': torch_net.conv[3].conv2,
        'conv4_3': torch_net.conv[3].conv3,
        'conv5_1': torch_net.conv[4].conv1,
        'conv5_2': torch_net.conv[4].conv2,
        'conv5_3': torch_net.conv[4].conv3,
        'fc6': torch_net.fc1[0],
        'fc7': torch_net.fc2[0],
        'fc8-101': torch_net.cls,
    }

    for k, layer in mappings.items():
        layer.weight.data.copy_(torch.from_numpy(caffe_params[k][0].data))
        layer.bias.data.copy_(torch.from_numpy(caffe_params[k][1].data))
    torch.save(torch_net, 'pth/age.pth')

def convert_gender():
    torch_net = Gender()

    caffe_net = caffe.Net('gender.prototxt', "gender.caffemodel", caffe.TEST)
    caffe_params = caffe_net.params

    mappings = {
        'conv1_1': torch_net.conv[0].conv1,
        'conv1_2': torch_net.conv[0].conv2,
        'conv2_1': torch_net.conv[1].conv1,
        'conv2_2': torch_net.conv[1].conv2,
        'conv3_1': torch_net.conv[2].conv1,
        'conv3_2': torch_net.conv[2].conv2,
        'conv3_3': torch_net.conv[2].conv3,
        'conv4_1': torch_net.conv[3].conv1,
        'conv4_2': torch_net.conv[3].conv2,
        'conv4_3': torch_net.conv[3].conv3,
        'conv5_1': torch_net.conv[4].conv1,
        'conv5_2': torch_net.conv[4].conv2,
        'conv5_3': torch_net.conv[4].conv3,
        'fc6': torch_net.fc1[0],
        'fc7': torch_net.fc2[0],
        'fc8-2': torch_net.cls,
    }

    for k, layer in mappings.items():
        layer.weight.data.copy_(torch.from_numpy(caffe_params[k][0].data))
        layer.bias.data.copy_(torch.from_numpy(caffe_params[k][1].data))
    torch.save(torch_net, 'pth/gender.pth')


if __name__ == '__main__':
    convert_age()
    convert_gender()