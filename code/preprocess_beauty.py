import os
import pickle
import argparse
from itertools import islice
import numpy as np
import pandas as pd
import random
import json
from collections import defaultdict
from sklearn import preprocessing

def load_interaction(data_dir):
    all_data = []
    f = open(data_dir, 'r')
    user_src2tgt_dict, item_src2tgt_dict = {}, {}
    for eachline in f:
        eachline = json.loads(eachline)
        u, i, r = eachline['reviewerID'], eachline['asin'], eachline['overall']
        if u not in user_src2tgt_dict:
            user_src2tgt_dict[u] = len(user_src2tgt_dict) + 1  # 从1编码
        user = user_src2tgt_dict[u]
        if i not in item_src2tgt_dict:
            item_src2tgt_dict[i] = len(item_src2tgt_dict) + 1
        item = item_src2tgt_dict[i]
        all_data.append([user, item, r])
    return all_data, user_src2tgt_dict, item_src2tgt_dict

def load_interaction2(data_dir):
    f = open(data_dir, 'r')

    user_count = {} #第一次过滤
    item_count = {}
    for eachline in f:
        eachline = json.loads(eachline)
        u, i, r = eachline['reviewerID'], eachline['asin'], eachline['overall']
        if u not in user_count:
            user_count[u] = 0
        user_count[u] += 1
        if i not in item_count:
            item_count[i] = 0
        item_count[i] += 1
    f.seek(0)

    user_new_count, item_new_count = {}, {} #第二次过滤
    for eachline in f:
        eachline = json.loads(eachline)
        u, i, r = eachline['reviewerID'], eachline['asin'], eachline['overall']
        if user_count[u] <= 7:
            continue
        if item_count[i] <= 7:
            continue

        if u not in user_new_count:
            user_new_count[u] = 0
        user_new_count[u] += 1
        if i not in item_new_count:
            item_new_count[i] = 0
        item_new_count[i] += 1

    f.seek(0)
    user_final_count, item_final_count = {}, {}
    all_data = []
    user_src2tgt_dict, item_src2tgt_dict = {}, {}
    for eachline in f:
        eachline = json.loads(eachline)
        u, i, r = eachline['reviewerID'], eachline['asin'], eachline['overall']
        if u not in user_new_count or user_new_count[u] < 5:
            continue
        if i not in item_new_count or item_new_count[i] < 5:
            continue
        if u not in user_src2tgt_dict:
            user_src2tgt_dict[u] = len(user_src2tgt_dict) + 1  # 从1编码
        user = user_src2tgt_dict[u]
        if i not in item_src2tgt_dict:
            item_src2tgt_dict[i] = len(item_src2tgt_dict) + 1
        item = item_src2tgt_dict[i]
        all_data.append([user, item, r])

        if user not in user_final_count:
            user_final_count[user] = 0
        user_final_count[user] += 1
        if item not in item_final_count:
            item_final_count[item] = 0
        item_final_count[item] += 1

    return all_data, user_src2tgt_dict, item_src2tgt_dict

def load_meta(f_meta, item_src2tgt_dict):
    cate_src2tgt_dict = {}
    cate_src_count = defaultdict(int)
    item_cate_dict = {}
    item_cate_dict[0] = 0
    f = open(f_meta, 'r', encoding='utf-8')
    for eachline in f:
        eachline = json.dumps(eval(eachline))
        eachline = json.loads(eachline)

        i, c = eachline['asin'], eachline['categories']
        if i in item_src2tgt_dict:
            for tmp_c in c:
                if tmp_c[0] == 'Beauty':
                    break;
            if len(tmp_c) >= 3:
                category = tmp_c[2]
            else:
                category = tmp_c[-1]
            #category = tmp_c[-1]
            if category not in cate_src2tgt_dict:
                cate_src2tgt_dict[category] = len(cate_src2tgt_dict) + 1
            cate = cate_src2tgt_dict[category]
            cate_src_count[category] += 1
            item_cate_dict[item_src2tgt_dict[i]] = cate   #item_id -> cate_id

    item_cate_dict = {k: v for k, v in sorted(item_cate_dict.items(), key=lambda kv: (kv[0], kv[1]))}
    return cate_src2tgt_dict, item_cate_dict

def create_user_list(all_data, user_size):
    user_list = [dict() for u in range(user_size + 1)]
    count = 0
    for u,i,w in all_data:
        user_list[u][i] = w
        count += 1
    print('interaction num:', count)
    return user_list

def split_train_test(user_list, test_size=0.2, time_order=False):
    train_user_list = [None] * len(user_list)
    test_user_list = [None] * len(user_list)
    for user, item_dict in enumerate(user_list):
        if time_order:
            # Choose latest item
            item = sorted(item_dict.items(), key=lambda x: x[1], reverse=True)
            latest_item = item[:int(len(item)*test_size)]
            assert max(item_dict.values()) == latest_item[0][1]
            test_item = set(map(lambda x: x[0], latest_item))
        else:
            # Random select
            test_item = set(np.random.choice(list(item_dict.keys()),
                                             size=int(len(item_dict)*test_size),
                                             replace=False))
        #if user>0:
        #    assert (len(test_item) > 0), "No test item for user %d" % user
        if user>0 and len(test_item)==0:
            print("No test item for user %d" % user)
        test_user_list[user] = test_item
        train_user_list[user] = set(item_dict.keys()) - test_item
    return train_user_list, test_user_list

def split_train_test_loo(user_list, test_size=0.2, time_order=False):
    train_user_list = [None] * len(user_list)
    test_user_list = [None] * len(user_list)
    for user, item_dict in enumerate(user_list):
        if time_order:
            # Choose latest item
            item = sorted(item_dict.items(), key=lambda x: x[1], reverse=True)
            latest_item = item[:int(len(item)*test_size)]
            assert max(item_dict.values()) == latest_item[0][1]
            test_item = set(map(lambda x: x[0], latest_item))
        else:
            # Random select
            test_item = set(np.random.choice(list(item_dict.keys()), min(1,len(item_dict))))
        #if user>0:
        #    assert (len(test_item) > 0), "No test item for user %d" % user
        if user>0 and len(test_item)==0:
            print("No test item for user %d" % user)
        test_user_list[user] = test_item
        train_user_list[user] = set(item_dict.keys()) - test_item
    return train_user_list, test_user_list



def create_pair(user_list):
    pair = []
    for user, item_set in enumerate(user_list):
        pair.extend([(user, item) for item in item_set])
    return pair


if __name__ == '__main__':
    random.seed(2020)
    np.random.seed(2020)
    f_interation = '../beauty/Beauty_5.json'
    f_meta = '../beauty/meta_Beauty.json'
    f_out = '../beauty/beauty_100neg.pkl'
    test_ratio = 0.2


    all_data, user_src2tgt_dict, item_src2tgt_dict = load_interaction2(f_interation)  # 加载数据，得到原id->正式id的dict
    user_size = len(user_src2tgt_dict)
    item_size = len(item_src2tgt_dict)

    cate_src2tgt_dict, item_cate_dict = load_meta(f_meta, item_src2tgt_dict)
    cate_size = len(cate_src2tgt_dict)
    print('user_num, item_num, cate_num', len(user_src2tgt_dict), len(item_src2tgt_dict), len(cate_src2tgt_dict))  # 5541, 3568, 60


    total_user_list = create_user_list(all_data, user_size) #把[u,i,r]转为list[u]={item...}
    print('avg. items per user: ', np.mean([len(u) for u in total_user_list]))  #11.68


    # ----------------------------------partition
    train_user_list, test_user_list = split_train_test(total_user_list,  # 划分train/test
                                                           test_size=test_ratio)
    print('min len(user_itemlist): ', min([len(ilist) for ilist in train_user_list[1:]]))

    train_pair = create_pair(train_user_list)
    print('Complete creating pair')


    dataset = {'user_size': user_size+1, 'item_size': item_size+1,
               'train_user_list': train_user_list, 'test_user_list': test_user_list,
               'cate_size': cate_size+1, 'item_cate_dict': item_cate_dict,
               'train_pair': train_pair}
    with open(f_out, 'wb') as f:
        #json.dump(dataset, f)
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)




