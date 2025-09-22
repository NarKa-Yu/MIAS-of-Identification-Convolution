import copy

import torch

from dataset import get_cifar10
from models import ResNet20, AlexNet, VGGNet, GoogLeNet3x3
import torch.nn as nn
import sys

from utility import get_attack_record, judge_exist_data

def attack_sgd(target_idx=0, batch_size=64, ic_type='midrange', region_type='var',
               amount=8, scaling_factor=100000):
    dataset_train, dataset_test, loader_train, loader_test = get_cifar10(batch_size=batch_size)
    # model = ResNet20()
    # model = AlexNet()
    # model = VGGNet()
    model = GoogLeNet3x3()
    target_data, target_label = dataset_test[target_idx]
    target_datas, target_labels = [target_data], [target_label]
    model, attack_records = get_attack_record(model=model, datas=target_datas,
                                              ic_type=ic_type, region_type=region_type,
                                              targets=target_labels, amount=amount, scaling_factor=scaling_factor)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    test_time, out_correct, in_correct = 0, 0, 0
    for idx, (data, label) in enumerate(loader_train):
        if idx % 100 == 0: print(f"idx = {idx}")
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, label)
        loss.backward()
        our_exist = judge_exist_data(conv_grad=model.conv1.weight.grad, data=target_datas[0],
                                    records=attack_records, threshold=1)
        # print(f"out is_exist = {our_exist}")
        test_time += 1
        if not our_exist:
            out_correct += 1

        data[0] = target_datas[0]
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, label)
        loss.backward()
        in_exist = judge_exist_data(conv_grad=model.conv1.weight.grad, data=target_datas[0],
                                    records=attack_records, threshold=1)
        # print(f"in is_exist = {in_exist}")
        test_time += 1
        if in_exist:
            in_correct += 1
    mia_accuracy = (in_correct + out_correct) / test_time
    print(f"MIA accuracy = {mia_accuracy}")
    print(f"in_correct = {in_correct} out_correct = {out_correct} test_time = {test_time}")
    return mia_accuracy

def attack_avg(target_idx=0, epoch=10, batch_size=64, ic_type='midrange', region_type='var',
               amount=8, scaling_factor=100000):
    dataset_train, dataset_test, loader_train, loader_test = get_cifar10(batch_size=batch_size)
    model = ResNet20()
    # model = AlexNet()
    # model = VGGNet()
    # model = GoogLeNet3x3()

    target_data, target_label = dataset_test[target_idx]
    target_datas, target_labels = [target_data], [target_label]
    model, attack_records = get_attack_record(model=model, datas=target_datas,
                                              ic_type=ic_type, region_type=region_type,
                                              targets=target_labels, amount=amount, scaling_factor=scaling_factor)

    conv1_weight = model.conv1.weight.data.detach().clone()  # 拷贝conv1的权重
    # print(f"conv1_weight = {conv1_weight}")
    params = model.state_dict()
    params = copy.deepcopy(params)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    test_time, out_correct, in_correct = 0, 0, 0
    for idx, (data, label) in enumerate(loader_train):
        if idx % 100 == 0: print(f"idx = {idx}")
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, label)
        loss.backward()
        optimizer.step()
        if (idx + 1) % epoch == 0:
            indirect_grad = conv1_weight - model.conv1.weight
            our_exist = judge_exist_data(conv_grad=indirect_grad, data=target_datas[0],
                                        records=attack_records, threshold=1)
            test_time += 1
            if not our_exist:
                out_correct += 1
            model.load_state_dict(params)

    model.load_state_dict(params) # 重置模型梯度
    for idx, (data, label) in enumerate(loader_train):
        if idx % 100 == 0: print(f"idx = {idx}")
        if (idx + 1) % epoch == 0:
            data[0] = target_datas[0] # 注入目标样本
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, label)
        loss.backward()
        optimizer.step()
        if (idx + 1) % epoch == 0:
            indirect_grad = conv1_weight - model.conv1.weight.grad
            in_exist = judge_exist_data(conv_grad=indirect_grad, data=target_datas[0],
                                         records=attack_records, threshold=1)
            test_time += 1
            if in_exist:
                in_correct += 1
            model.load_state_dict(params)
    mia_accuracy = (in_correct + out_correct) / test_time
    print(f"MIA accuracy = {mia_accuracy}")
    print(f"in_correct = {in_correct} out_correct = {out_correct} test_time = {test_time}")
    return mia_accuracy

def analysis_args():
    params_dict = {}
    args = sys.argv
    for arg in args:
        idx = arg.find('=')
        if idx != -1:
            key = arg[:idx]
            value = arg[idx+1:]
            if value.isdigit():
                value = int(value)
            params_dict[key] = value
    return params_dict

if __name__ == "__main__":
    params_dict = analysis_args()

    if params_dict['fed'] == 'avg':
        attack_avg(target_idx=params_dict['idx'], epoch=params_dict['epoch'], batch_size=params_dict['batch_size'],
                   ic_type=params_dict['ic'], region_type=params_dict['region'], amount=params_dict['amount'],
                   scaling_factor=params_dict['scaling'])
    elif params_dict['fed'] == 'sgd':
        attack_sgd(target_idx=params_dict['idx'], batch_size=params_dict['batch_size'],
                   ic_type=params_dict['ic'], region_type=params_dict['region'], amount=params_dict['amount'],
                   scaling_factor=params_dict['scaling'])
    # python attack.py fed=sgd idx=0 amount=8 scaling=10000 epoch=10 batch_size=64 ic=midrange region=var

    1/0
    # x = torch.randn(3, 3, 3)
    # ic = get_mean_ic(input_tensor=x)
    # test_xs = torch.randn(1000, 3, 3, 3)
    # test_xs = ic * test_xs
    # print(f"test_xs = {torch.sum(test_xs, dim=(1, 2, 3))}")
