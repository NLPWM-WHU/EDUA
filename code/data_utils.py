import torch
import random
import pandas as pd
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from _collections import defaultdict
import time


def generate_user_diverse(train_user_list, item_cate_dict):
    user_freq = []
    for u, ilist in enumerate(train_user_list):
        cate_set = set()
        tmp_cate_dict = defaultdict(int)
        for i in ilist:
            cate_set.add(item_cate_dict[i])
            tmp_cate_dict[item_cate_dict[i]] += 1
        tmp = 0
        for k, v in tmp_cate_dict.items():
            tmp += 1 / v
        user_freq.append(tmp / (len(tmp_cate_dict) + 1e-6))
        #user_freq.append(0.4 * len(cate_set) / (len(ilist) + 1e-5) + 0.6* len(cate_set)/cate_num)   #考虑用户买的类别数占用户买的item数越多（越多样）、用户买的类别数占总类别数越多（越多样）
    user_freq[0] = 0.5
    user_freq = np.array(user_freq)
    min, max = user_freq.min(), user_freq.max()
    k = (1-0.) / (max - min)
    user_freq = 0. + k * (user_freq - min)   #归一化到[0,0.5]

    return user_freq

def generate_randwalk(user_i_list, item_c_list, train_pair, item_num):
    """
    负采样，一部分由unclicked中随机采，一部分构造random walk路径从unclicked采；
    :param user_i_list:
    :param item_c_list:
    :param train_pair:
    :param item_num:
    :return:
    """
    start = time.time()
    cate_i_list = defaultdict(list)
    for i, c in enumerate(item_c_list):
        cate_i_list[c].append(i)

    item_u_list = [[] for i in range(item_num)]
    for u, i in train_pair:
        item_u_list[i].append(u)
    for i in range(len(item_u_list)):
        if len(item_u_list[i]) == 0:
            #print(i)
            item_u_list[i] = np.random.choice(range(1, len(user_i_list)), size=1).tolist()

    user_i_list[0] = set([0])
    item_u_list[0] = [0]

    walk_length = 5
    walk_num = 10
    i_rwlist = []
    for i in range(len(item_c_list)):
        i_tmp_list = []
        for n in range(walk_num):
            curr_i = i
            for l in range(walk_length):    #ICIUI
                curr_c = item_c_list[curr_i]
                curr_i = np.random.choice(cate_i_list[curr_c])
                i_tmp_list.append(curr_i)

                curr_u = item_u_list[curr_i]
                curr_u = np.random.choice(curr_u)
                curr_i = list(user_i_list[curr_u])
                curr_i = np.random.choice(curr_i)
                i_tmp_list.append(curr_i)

            curr_i = i
            for l in range(walk_length):    #IUICI
                curr_u = item_u_list[curr_i]
                curr_u = np.random.choice(curr_u)
                curr_i = list(user_i_list[curr_u])
                curr_i = np.random.choice(curr_i)
                i_tmp_list.append(curr_i)

                curr_c = item_c_list[curr_i]
                curr_i = np.random.choice(cate_i_list[curr_c])
                i_tmp_list.append(curr_i)

        i_rwlist.append(i_tmp_list)

    return i_rwlist

class UserItemRatingDataset_pair(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, negs_tensor, reverse_item_tensor, reverse_negs_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.negs_tensor = negs_tensor
        self.reverse_item_tensor = reverse_item_tensor
        self.reverse_negs_tensor = reverse_negs_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.negs_tensor[index], self.reverse_item_tensor[index], self.reverse_negs_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


def instance_a_train_loader(user_num, item_num, train_pair, train_user_list, item_cate_arr, cate_item_dict, num_negatives, batch_size, user_alpha):
    #instance train loader for one training epoch

    #start = time.time()
    item_rw_list = generate_randwalk(train_user_list.copy(), item_cate_arr.copy(), train_pair, item_num)

    users, items, negs = [], [], []
    reverse_items, reverse_negs = [], []
    for u, ilist in enumerate(train_user_list):
        # print(u)

        ilist = list(ilist)
        if len(ilist) == 0:
            continue
        clist = item_cate_arr[ilist]
        cate_weight, cate_item = dict(), dict()
        for item, cate in zip(ilist, clist):
            if cate not in cate_weight:
                cate_weight[cate] = 0
            if cate not in cate_item:
                cate_item[cate] = []
            cate_weight[cate] += 1
            cate_item[cate].append(item)
        max_weight = max(cate_weight.values())
        for cate, weight in cate_weight.items():
            cate_weight[cate] = max_weight / weight
        sum_weight = sum(cate_weight.values())
        for cate, weight in cate_weight.items():
            cate_weight[cate] = weight / sum_weight

        #reversed positive items
        samp_cate_list = np.random.choice(list(cate_weight.keys()), size=len(ilist), replace=True, p=list(cate_weight.values()))
        for idx, cate in enumerate(samp_cate_list):
            if len(cate_item[cate]) < 2:
                i = np.random.choice(cate_item_dict[cate])
            else:
                i = np.random.choice(cate_item[cate])

            if np.random.rand() >= user_alpha[u]:    #user alpha表示准确性的关注，如果很低说明按照原始ilist[idx]
                i = ilist[idx]
            else:
                i = i
            reverse_items.append(int(i))

            #negative sampling
            tmp_neg = []
            curr_i_list = item_rw_list[i]
            for _ in range(num_negatives):
                if np.random.rand() < 0.8:  # best = 0.8
                    neg_i = np.random.randint(1, item_num)
                    while neg_i in ilist:
                        neg_i = np.random.randint(1, item_num)
                    tmp_neg.append(neg_i)
                else:
                    neg_i = np.random.choice(curr_i_list)
                    while neg_i in ilist:
                        neg_i = np.random.choice(curr_i_list)
                    tmp_neg.append(neg_i)
            reverse_negs.append(tmp_neg)

        #uniform positive items
        for i in ilist:
            users.append(u)
            items.append(int(i))

            # negative sampling
            tmp_neg = []
            curr_i_list = item_rw_list[i]
            for _ in range(num_negatives):
                if np.random.rand() < 0.8: #best = 0.8
                    neg_i = np.random.randint(1, item_num)
                    while neg_i in ilist:
                        neg_i = np.random.randint(1, item_num)
                    tmp_neg.append(neg_i)
                else:
                    neg_i = np.random.choice(curr_i_list)
                    while neg_i in ilist:
                        neg_i = np.random.choice(curr_i_list)
                    tmp_neg.append(neg_i)
            negs.append(tmp_neg)
    #end = time.time()
    #print('sample time: ', end - start)

    dataset = UserItemRatingDataset_pair(user_tensor=torch.LongTensor(np.array(users)),
                                         item_tensor=torch.LongTensor(np.array(items)),
                                         negs_tensor=torch.LongTensor(np.array(negs)),
                                         reverse_item_tensor=torch.LongTensor(np.array(reverse_items)),
                                         reverse_negs_tensor=torch.LongTensor(np.array(reverse_negs)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
