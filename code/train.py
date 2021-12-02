import pandas as pd
import numpy as np
import random
import time
import pickle
from _collections import defaultdict
import argparse
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from data_utils import instance_a_train_loader
from metric import getR, getNDCG, calculate_ndcg, getCC, getILD

from model import EDUA

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


parser = argparse.ArgumentParser()

# Train Test
parser.add_argument('--n_epochs',
                    type=int,
                    default=20,
                    help="Number of epoch during training")
parser.add_argument('--batch_size',
                    type=int,
                    default=128,
                    help="Batch size in one iteration")
parser.add_argument('--num_neg',
                    type=int,
                    default=20,
                    help="Number of Negative Sampling")
# Model
parser.add_argument('--num_memory',
                    type=int,
                    default=10)
parser.add_argument('--dim',
                    type=int,
                    default=50,
                    help="Dimension for embedding")
# Optimizer
parser.add_argument('--lr',
                    type=float,
                    default=0.0005,
                    help="Learning rate")
parser.add_argument('--margin',
                    type=float,
                    default=1.0,
                    help="Margin loss")
parser.add_argument('--weight_decay',
                    type=float,
                    default=0,
                    help="Weight decay factor")
args = parser.parse_args()


def evaluate(user_pred, test_user_list, train_user_list):
    lst_hr_5, lst_hr_10, lst_ndcg_5, lst_ndcg_10 = [], [], [], []
    lst_cc_5, lst_cc_10, lst_ild_5, lst_ild_10 = [], [], [], []
    tmp_ratio_5, tmp_ratio_10 = [], []
    item_cate_arr = np.array(list(item_cate_dict.values()))
    for u in user_pred.keys():
        if len(test_user_list[u]) == 0 or len(train_user_list[u]) == 0:
            continue
        u_pred = sorted(user_pred[u], key=lambda x:x[1], reverse=True)
        u_pred_5 = [u[0] for u in u_pred[:5]]
        u_pred_10 = [u[0] for u in u_pred[:10]]
        hr_5, ndcg_5 = getR(u_pred_5, test_user_list[u]), calculate_ndcg(test_user_list[u], u_pred_5, 10)#getNDCG(u_pred_5, test_user_list[u])
        cc_5, ild_5 = getCC(cate_num, u_pred_5, test_user_list[u], item_cate_dict), getILD(u_pred_5, item_cate_dict)
        hr_10, ndcg_10 = getR(u_pred_10, test_user_list[u]), calculate_ndcg(test_user_list[u], u_pred_10, 20)#getNDCG(u_pred_10, test_user_list[u])
        cc_10, ild_10 = getCC(cate_num, u_pred_10, test_user_list[u], item_cate_dict), getILD(u_pred_10, item_cate_dict)
        lst_hr_5.append(hr_5)
        lst_hr_10.append(hr_10)
        lst_ndcg_5.append(ndcg_5)
        lst_ndcg_10.append(ndcg_10)
        lst_cc_5.append(cc_5)
        lst_cc_10.append(cc_10)
        lst_ild_5.append(ild_5)
        lst_ild_10.append(ild_10)

        tmp_ratio_5.append(len(set(item_cate_arr[u_pred_5])) / 5)
        tmp_ratio_10.append(len(set(item_cate_arr[u_pred_10])) / 10)

    partition = 10
    min, max = 0.0, 1.0
    gap = (max - min) / partition
    part_count = [0 for i in range(10)]
    for ratio in tmp_ratio_5:
        for p in range(partition):
            if ratio > min + p * gap and ratio <= min + (p + 1) * gap:
                part_count[p] += 1
                break
    print(part_count)
    part_count = [0 for i in range(10)]
    for ratio in tmp_ratio_10:
        for p in range(partition):
            if ratio > min + p * gap and ratio <= min + (p + 1) * gap:
                part_count[p] += 1
                break
    print(part_count)

    return sum(lst_hr_5)/len(lst_hr_5), sum(lst_ndcg_5)/len(lst_ndcg_5), sum(lst_hr_10)/len(lst_hr_10), sum(lst_ndcg_10)/len(lst_ndcg_10), sum(lst_cc_5)/len(lst_cc_5), sum(lst_ild_5)/len(lst_ild_5), sum(lst_cc_10)/len(lst_cc_10), sum(lst_ild_10)/len(lst_ild_10)

def statistic(tmp_ratio_5, tmp_ratio_10):
    partition = 10
    min, max = 0.0, 1.0
    gap = (max - min) / partition
    part_count = [0 for i in range(10)]
    for ratio in tmp_ratio_5:
        for p in range(partition):
            if ratio > min + p * gap and ratio <= min + (p + 1) * gap:
                part_count[p] += 1
                break
    print(part_count)
    part_count = [0 for i in range(10)]
    for ratio in tmp_ratio_10:
        for p in range(partition):
            if ratio > min + p * gap and ratio <= min + (p + 1) * gap:
                part_count[p] += 1
                break
    print(part_count)
    return


def generate_user_aspect(user_cate_list):
    user_cat_mat = np.zeros([user_num, cate_num])
    for u, c_list in user_cate_list.items():
        for c in c_list:
            user_cat_mat[u][c] += 1
        # user_cat_mat[u] = user_cat_mat[u] / sum(user_cat_mat[u])
    Scaler = preprocessing.MinMaxScaler().fit(user_cat_mat)
    user_cat_mat = Scaler.transform(user_cat_mat)

    #tsne = TSNE(n_components=args.num_memory, init='pca', method='exact')
    tsne = PCA(n_components=args.num_memory)
    Y = tsne.fit_transform(user_cat_mat)  # 转换后的输出
    Scaler2 = preprocessing.MinMaxScaler().fit(Y)
    Y = Scaler2.transform(Y)
    return Y

def generate_cate_aspect(train_pair, item_cate_dict):
    cate_user_mat = np.zeros([cate_num, user_num])
    for u, i in train_pair:
        c = item_cate_dict[i]
        cate_user_mat[c][u] += 1
    Scaler = preprocessing.MinMaxScaler().fit(cate_user_mat)
    cate_user_mat = Scaler.transform(cate_user_mat)
    #tsne = TSNE(n_components=args.num_memory, init='pca', method='exact')
    tsne = PCA(n_components=args.num_memory)
    Y = tsne.fit_transform(cate_user_mat)  # 转换后的输出
    Scaler2 = preprocessing.MinMaxScaler().fit(Y)
    Y = Scaler2.transform(Y)
    return Y

def statistic_user_div_score(train_user_list, item_cate_dict):
    # 为了统计数据集中，不同用户是否热衷于concentrated items/diverse items，计算每个用户 Topic Coverage(TC)/#item list
    user_cate_div_item_ratio = {}
    for user, item_dict in enumerate(train_user_list):
        itemlist = list(item_dict)

        tmp_cate_set = set()
        for i in itemlist:
            c = item_cate_dict[i]
            tmp_cate_set.add(c)
        user_cate_div_item_ratio[user] = len(tmp_cate_set)/ (len(itemlist)+1e-6)

    return user_cate_div_item_ratio


if __name__ == '__main__':
    #random.seed(2020)
    #np.random.seed(2020)

    data_set = 'music'
    f_data = '../{}/{}_data.pkl'.format(data_set, data_set)
    max_len = {'music': 0.995, 'beauty': 0.995, 'ml1m': 0.8}
    if data_set == 'ml1m':
        args.n_epochs = 10
        args.n_batch_size = 256
        args.lr = 0.00025
    print(data_set, args)

    with open(f_data, 'rb') as f:
        dataset = pickle.load(f)
        user_num, item_num, cate_num = dataset['user_size'], dataset['item_size'], dataset['cate_size']
        item_cate_dict = dataset['item_cate_dict']
        train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']
        train_pair = dataset['train_pair']

        train_item_list = [[] for i in range(item_num)]
        for u,i in train_pair:
            train_item_list[i].append(u)
        for i in range(len(train_item_list)):   #ml1m存在无交互item
            if len(train_item_list[i]) == 0:
                train_item_list[i] = np.random.choice(range(1,user_num), size=1).tolist()

        cate_item_dict = {}
        for i,c in item_cate_dict.items():
            if c not in cate_item_dict:
                cate_item_dict[c] = []
            cate_item_dict[c].append(i)

        item_cate_arr = np.array(list(item_cate_dict.values()))
        user_cate_list = {}
        for u, items in enumerate(train_user_list):
            items = list(items)
            cates = item_cate_arr[items]
            user_cate_list[u] = list(cates)

    print(user_num, item_num, cate_num)
    print('Load complete')

    user_aspect_arr, cate_aspect_arr = generate_user_aspect(user_cate_list), generate_cate_aspect(train_pair, item_cate_dict)

    user_div_score = np.array(list(statistic_user_div_score(train_user_list, item_cate_dict).values()))
    max_len_item = sorted([len(lst) for lst in train_user_list])[int(max_len[data_set] * user_num)]
    max_len_user = sorted([len(lst) for lst in train_item_list])[int(max_len[data_set] * item_num)]
    model = EDUA(user_num, item_num, cate_num, args.dim, args.margin, args.num_memory, max_len_item, max_len_user)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #torch.nn.utils.clip_grad_norm(model.parameters(), 1)

    for epoch in range(0, args.n_epochs):
        print('#' * 80)
        print('Epoch {} starts !'.format(epoch))
        start_t = time.time()

        train_loader = instance_a_train_loader(user_num, item_num, train_pair, train_user_list, item_cate_arr, cate_item_dict, args.num_neg, args.batch_size, user_div_score)

        model.train()
        total_loss = 0

        if data_set == 'ml1m':
            alpha = 1 - np.array([epoch / (2 * args.n_epochs)]).repeat(user_num)
        else:
            alpha = np.array([epoch / (2 * args.n_epochs)]).repeat(user_num)
        #--------------------------train
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, neg_item, reverse_item, reverse_neg = batch[0], batch[1], batch[2], batch[3], batch[4]
            tmp_alpha = torch.FloatTensor(alpha[user.numpy()]).cuda()

            u_div_score = torch.FloatTensor(user_div_score[user.numpy()]).cuda()

            u_item_list = []
            for u in user:
                u_item_list.append(list(train_user_list[int(u.numpy())]))
            for idx, c in enumerate(u_item_list):
                if len(c) < max_len_item:
                    c += [0] * (max_len_item - len(c))
                else:
                    c = c[:max_len_item]
                u_item_list[idx] = c
            u_item_list = torch.LongTensor(np.array(u_item_list))

            i_user_list = []
            for i in item:
                i_user_list.append(list(train_item_list[int(i.numpy())]))
            for idx, c in enumerate(i_user_list):
                if len(c) < max_len_user:
                    c += [0] * (max_len_user - len(c))
                else:
                    c = c[:max_len_user]
                i_user_list[idx] = c
            i_user_list = torch.LongTensor(np.array(i_user_list))

            neg_user_list = []
            for neg in range(neg_item.shape[1]):
                tmp_user_list = []
                for i in neg_item[:, neg]:
                    tmp_user_list.append(list(train_item_list[int(i.numpy())]))
                for idx, c in enumerate(tmp_user_list):
                    if len(c) < max_len_user:
                        c += [0] * (max_len_user - len(c))
                    else:
                        c = c[:max_len_user]
                    tmp_user_list[idx] = c
                tmp_user_list = torch.LongTensor(np.array(tmp_user_list)).cuda()
                neg_user_list.append(tmp_user_list)

            rev_i_user_list = []
            for i in reverse_item:
                rev_i_user_list.append(list(train_item_list[int(i.numpy())]))
            for idx, c in enumerate(rev_i_user_list):
                if len(c) < max_len_user:
                    c += [0] * (max_len_user - len(c))
                else:
                    c = c[:max_len_user]
                rev_i_user_list[idx] = c
            rev_i_user_list = torch.LongTensor(np.array(rev_i_user_list))

            rev_neg_user_list = []
            for neg in range(reverse_neg.shape[1]):
                tmp_user_list = []
                for i in reverse_neg[:, neg]:
                    tmp_user_list.append(list(train_item_list[int(i.numpy())]))
                for idx, c in enumerate(tmp_user_list):
                    if len(c) < max_len_user:
                        c += [0] * (max_len_user - len(c))
                    else:
                        c = c[:max_len_user]
                    tmp_user_list[idx] = c
                tmp_user_list = torch.LongTensor(np.array(tmp_user_list)).cuda()
                rev_neg_user_list.append(tmp_user_list)

            #-----------aspect
            user_aspect = torch.FloatTensor(user_aspect_arr[user.numpy()])
            item_cate_tmp = item_cate_arr[item.numpy()]
            item_aspect = torch.FloatTensor(cate_aspect_arr[item_cate_tmp])
            neg_item_cate_tmp = item_cate_arr[neg_item.numpy()]
            neg_item_aspect = torch.FloatTensor(cate_aspect_arr[neg_item_cate_tmp])
            reverse_item_cate_tmp = item_cate_arr[reverse_item.numpy()]
            reverse_item_aspect = torch.FloatTensor(cate_aspect_arr[reverse_item_cate_tmp])
            reverse_neg_cate_tmp = item_cate_arr[reverse_neg.numpy()]
            reverse_neg_aspect = torch.FloatTensor(cate_aspect_arr[reverse_neg_cate_tmp])

            optimizer.zero_grad()

            loss = model(user.cuda(), item.cuda(), neg_item.cuda(), reverse_item.cuda(), reverse_neg.cuda(), u_item_list.cuda(), i_user_list.cuda(), neg_user_list, rev_i_user_list.cuda(), rev_neg_user_list, tmp_alpha, user_aspect.cuda(), item_aspect.cuda(), neg_item_aspect.cuda(), reverse_item_aspect.cuda(), reverse_neg_aspect.cuda(), u_div_score.cuda())
            loss.backward()
            optimizer.step()
            #print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch, batch_id, loss.data[0]))
            #print(loss.data[0])
            total_loss += loss.item()

        print( 'loss: %f' % (total_loss / (batch_id+1)))


        #--------------------------test
        if (epoch) % 1 == 0:
            model.eval()
            model.zero_grad()
            lst_hr_5, lst_hr_10, lst_ndcg_5, lst_ndcg_10 = [], [], [], []
            lst_cc_5, lst_cc_10, lst_ild_5, lst_ild_10 = [], [], [], []
            tmp_ratio_5, tmp_ratio_10 = [], []
            item_cate_arr = np.array(list(item_cate_dict.values()))

            item = torch.LongTensor(np.array(range(item_num)))
            i_user_list = []
            for i in item:
                i_user_list.append(list(train_item_list[int(i.numpy())]))
            #max_len_user = max(len(c) for c in i_user_list)
            for idx, c in enumerate(i_user_list):
                if len(c) < max_len_user:
                    c += [0] * (max_len_user - len(c))
                else:
                    c = c[:max_len_user]
                i_user_list[idx] = c
            i_user_list = torch.LongTensor(np.array(i_user_list))

            for u in range(1, user_num):
                if len(test_user_list[u]) == 0:
                    continue

                pred = []
                user = torch.LongTensor(np.array([u]))
                u_item_list = []
                for u in user:
                    u_item_list.append(list(train_user_list[int(u.numpy())]))

                for idx, c in enumerate(u_item_list):
                    if len(c) < max_len_item:
                        c += [0] * (max_len_item - len(c))
                    else:
                        c = c[:max_len_item]
                    u_item_list[idx] = c
                u_item_list = torch.LongTensor(np.array(u_item_list))

                user_aspect = torch.FloatTensor(user_aspect_arr[user.numpy()])
                item_cate_tmp = item_cate_arr[item.numpy()]
                item_aspect = torch.FloatTensor(cate_aspect_arr[item_cate_tmp])

                ratings_pred = model.predict(user.cuda(), item.cuda(), u_item_list.cuda(), i_user_list.cuda(), 0.5, user_aspect.cuda(), item_aspect.cuda())
                ratings_pred = (ratings_pred.view(-1)).cpu().detach().numpy()

                idx = np.zeros_like(ratings_pred, dtype=bool)
                for i_idx in train_user_list[u]:
                    idx[i_idx] = True
                ratings_pred[idx] = -np.inf
                ratings_pred[0] = -np.inf
                pred = np.argsort(-ratings_pred)

                u_pred_5 = pred[:5]
                u_pred_10 = pred[:10]
                hr_5, ndcg_5 = getR(u_pred_5, test_user_list[u]), calculate_ndcg(test_user_list[u], u_pred_5, 5)
                cc_5, ild_5 = getCC(cate_num, u_pred_5, test_user_list[u], item_cate_dict), getILD(u_pred_5, item_cate_dict)
                hr_10, ndcg_10 = getR(u_pred_10, test_user_list[u]), calculate_ndcg(test_user_list[u], u_pred_10, 10)
                cc_10, ild_10 = getCC(cate_num, u_pred_10, test_user_list[u], item_cate_dict), getILD(u_pred_10, item_cate_dict)
                lst_hr_5.append(hr_5)
                lst_hr_10.append(hr_10)
                lst_ndcg_5.append(ndcg_5)
                lst_ndcg_10.append(ndcg_10)
                lst_cc_5.append(cc_5)
                lst_cc_10.append(cc_10)
                lst_ild_5.append(ild_5)
                lst_ild_10.append(ild_10)

                tmp_ratio_5.append(len(set(item_cate_arr[u_pred_5])) / 5)
                tmp_ratio_10.append(len(set(item_cate_arr[u_pred_10])) / 10)

            print('recall/hr_5=%f, ndcg_5=%f, cc_5=%f, ild_5=%f, recall/hr_10=%f, ndcg_10=%f, cc_10=%f, ild_10=%f' % (sum(lst_hr_5) / len(lst_hr_5), sum(lst_ndcg_5) / len(lst_ndcg_5), sum(lst_cc_5) / len(lst_cc_5), sum(lst_ild_5) / len(lst_ild_5), sum(lst_hr_10) / len(lst_hr_10), sum(lst_ndcg_10) / len(lst_ndcg_10), sum(lst_cc_10) / len(lst_cc_10), sum(lst_ild_10) / len(lst_ild_10)))
            statistic(tmp_ratio_5, tmp_ratio_10)
