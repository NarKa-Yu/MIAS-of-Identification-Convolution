import torch

import torch.nn as nn
import torch.nn.functional as F

# 通过最大最小值的均值构造mid midrange(中程数)表示最大值和最小值的均值 反映了数据范围的中心位置
def get_midrange_ic(input_tensor, scaling_factor = 100):
    '''
    构造识别卷积：
    根据卷积区域的中程数计算中间值 然后计算较大值数量和较小值数量
    需要注意的是 0元素无法作为分母 卷积区域内的0元素使用的是零值约束 不包含在最大最小值约束内
    所以如果0元素被包含在较大值或者较小值内时 需要从较大值或较小值数量上减掉0元素数量
    '''
    shape = input_tensor.shape
    flatten_tensor = input_tensor.flatten()
    clone_tensor = flatten_tensor.detach().clone()  # 先分离计算图，再克隆
    max_element = torch.max(flatten_tensor)
    min_element =  torch.min(flatten_tensor)
    mid = (max_element + min_element) / 2
    # print(f"(flatten_tensor >= mid) = {(flatten_tensor >= mid)}")
    bigger_num = torch.sum(flatten_tensor >= mid)
    smaller_num = torch.sum(flatten_tensor < mid)
    zeros_num = torch.sum(flatten_tensor == 0)
    if mid <= 0:
        bigger_num -= zeros_num # 从较大值数量中减掉0元素数量
    else:
        smaller_num -= zeros_num # 从较小值数量中减掉0元素数量
    # print(f"bigger_num = {bigger_num} smaller_num = {smaller_num}")
    # print(f"shape = {shape} flatten_tensor.shape = {flatten_tensor.shape}")
    for i, (num) in enumerate(flatten_tensor):
        if num == 0:
            clone_tensor[i] = scaling_factor
        elif num >= mid:
            clone_tensor[i] = 1 / (num * bigger_num) * scaling_factor
        else:
            clone_tensor[i] = -1 / (num * smaller_num) * scaling_factor
    result = torch.sum(flatten_tensor * clone_tensor, dim=0)
    # print(f'result = {result}')
    return clone_tensor.reshape(shape)

# 通过整体均值构造mid
def get_mean_ic(input_tensor, scaling_factor = 100):
    shape = input_tensor.shape
    flatten_tensor = input_tensor.flatten()
    clone_tensor = flatten_tensor.detach().clone()  # 先分离计算图，再克隆
    mid = torch.mean(flatten_tensor)
    bigger_num = torch.sum(flatten_tensor >= mid)
    smaller_num = torch.sum(flatten_tensor < mid)
    zeros_num = torch.sum(flatten_tensor == 0)
    if mid <= 0:
        bigger_num -= zeros_num # 从较大值数量中减掉0元素数量
    else:
        smaller_num -= zeros_num # 从较小值数量中减掉0元素数量
    for i, (num) in enumerate(flatten_tensor):
        if num == 0:
            clone_tensor[i] = scaling_factor
        elif num >= mid:
            clone_tensor[i] = 1 / (num * bigger_num) * scaling_factor
        else:
            clone_tensor[i] = -1 / (num * smaller_num) * scaling_factor
    return clone_tensor.reshape(shape)

# 根据最大最小值的差值大小来给卷积区域做选择优先级 越大优先级越高
def get_gap_region(data, amount, shape=(3, 3, 3), padding=1, kernel_size=3, stride=1):
    windows = F.unfold(data, kernel_size=kernel_size, stride=stride, padding=padding)
    maxs, _ = torch.max(windows, dim=0)
    mins, _ = torch.min(windows, dim=0)
    gaps = maxs - mins
    gaps_indices = torch.argsort(gaps, dim=0, descending=True) # 获得从大到小的索引(原tensor索引 没有进行排序)
    fit_indices = gaps_indices[: amount]
    fit_windows = windows[:, fit_indices]
    regions = []
    for i in range(amount):
        region = fit_windows[:, i].reshape(shape)
        regions.append(region)
    return regions


# 根据卷积区域内的元素总方差确定优先级(离散程度) 越大优先级越高
def get_var_region(data, amount, shape=(3, 3, 3), padding=1, kernel_size=3, stride=1):
    windows = F.unfold(data, kernel_size=kernel_size, stride=stride, padding=padding)
    vars, means = torch.var_mean(windows, dim=0)
    vars_indices = torch.argsort(vars, dim=0, descending=True)  # 获得从大到小的索引(原tensor索引 没有进行排序)
    fit_indices = vars_indices[: amount]
    fit_windows = windows[:, fit_indices]
    regions = []
    for i in range(amount):
        region = fit_windows[:, i].reshape(shape)
        regions.append(region)
    return regions

# 获取攻击记录
def get_attack_record(model, datas, targets, amount, shape=(3, 3, 3), scaling_factor=10000, padding=0,
                      ic_type='midrange', region_type='var'):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    records = []
    for i in range(len(datas)):
        optimizer.zero_grad()
        data, target = datas[i], targets[i]
        target = torch.tensor(target, dtype=torch.long)
        target = target.unsqueeze(dim=0)
        test_data = data.unsqueeze(dim=0)
        ics, regions = [], []
        # 对data按制定方法获取指定数量的识别卷积
        if region_type == 'gap':
            regions = get_gap_region(data, amount=amount, shape=shape, padding=padding)
        elif region_type == 'var':
            regions = get_var_region(data, amount=amount, shape=shape, padding=padding)
        for region in regions:
            if ic_type == 'midrange':
                ic = get_midrange_ic(region, scaling_factor=scaling_factor)
            elif ic_type == 'mean':
                ic = get_mean_ic(region, scaling_factor=scaling_factor)
            ics.append(ic)
        # print(f"model.conv1.weight.data.shape = {model.conv1.weight.data.shape}")
        # 在模型的第一层卷积层安装识别卷积并记录梯度情况
        for j, (ic_tensor) in enumerate(ics):
            record = {}
            cur_index = i * amount + j
            with torch.no_grad():
                model.conv1.weight.data[cur_index, :, :, :] = ic_tensor
            optimizer.zero_grad()
            output = model(test_data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            cur_grad = model.conv1.weight.grad[cur_index, :, :, :]
            # print(f"cur_grad = {cur_grad}")
            record['index'] = cur_index
            record['ic'] = ic_tensor
            record['data'] = data
            record['grad'] = cur_grad
            records.append(record)
    return model, records

# 根据对应梯度是否存在判断是否存在目标样本
def judge_exist_data(conv_grad, data, records, threshold=0.999):
    # print(f"conv_grad.shape = {conv_grad.shape} data.shape = {data.shape}")
    record_num, target_num = 0, 0
    for i, (record) in enumerate(records):
        # print(f"record[data] = {record['data'] }")
        is_equal = torch.sum((record['data'] == data) == False) == 0
        # print(f"is_equal = {is_equal} i = {i}")
        if is_equal:
            record_grad = record['grad']
            grad_idx = record['index']
            # print(f"grad_idx = {grad_idx}")
            target_grad = conv_grad[grad_idx, :, :, :]
            record_grad = record_grad.flatten()
            target_grad = target_grad.flatten()
            # print(f"record_grad.shape = {record_grad.shape}")
            # print(f"target_grad.shape = {target_grad.shape}")
            for cur_idx, (grad) in enumerate(record_grad):
                if record_grad[cur_idx] != 0:
                    record_num += 1
                    if target_grad[cur_idx] != 0:
                        target_num += 1
    judge_ratio = target_num / record_num
    # print(f"judge_ratio = {judge_ratio}")
    # print(f"target_num = {target_num} record_num = {record_num}")
    if judge_ratio >= threshold:
        return True
    else:
        return False
