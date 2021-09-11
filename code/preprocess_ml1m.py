import numpy as np
import scipy.sparse as sp
import pickle
import os
import pandas as pd
from _collections import defaultdict
import random
from sklearn import preprocessing

def load_interaction(data_dir):
    all_data = []
    f = open(data_dir, 'r')
    user_src2tgt_dict, item_src2tgt_dict = {}, {}
    for eachline in f:
        #eachline = json.loads(eachline)
        #u, i, r = eachline['reviewerID'], eachline['asin'], eachline['overall']
        eachline = eachline.split('::')
        u, i, r, _ = int(eachline[0]), int(eachline[1]), int(eachline[2]), int(eachline[3])
        if u not in user_src2tgt_dict:
            user_src2tgt_dict[u] = len(user_src2tgt_dict) + 1  # 从1编码
        user = user_src2tgt_dict[u]
        if i not in item_src2tgt_dict:
            item_src2tgt_dict[i] = len(item_src2tgt_dict) + 1
        item = item_src2tgt_dict[i]
        all_data.append([user, item, r])
    return all_data, user_src2tgt_dict, item_src2tgt_dict

def create_user_list(all_data, user_size):
    user_list = [dict() for u in range(user_size + 1)]
    count = 0
    for u,i,w in all_data:
        user_list[u][i] = w
        count += 1
    print('interaction num:', count)
    return user_list

def create_pair(user_list):
    pair = []
    for user, item_set in enumerate(user_list):
        pair.extend([(user, item) for item in item_set])
    return pair

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


if __name__ == '__main__':
    random.seed(2020)
    np.random.seed(2020)
    test_ratio = 0.2

    f_userinfo = '../ml1m/users.dat'
    f_iteminfo = '../ml1m/movies.dat'
    f_dataall = '../ml1m/ratings.dat'
    f_out = '../ml1m/ml1m.pkl'

    #load interaction
    all_data, user_src2tgt_dict, item_src2tgt_dict = load_interaction(f_dataall)
    user_size = len(user_src2tgt_dict)
    item_size = len(item_src2tgt_dict)

    # ------------------------------------------------load item
    cate_list = ["Unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    cate_src2tgt_dict = {f: i for i, f in enumerate(cate_list)}
    genre_count = defaultdict(list)
    item_cate_dict = {}
    item_cate_dict[0] = 0
    movie_df = pd.read_csv(f_iteminfo, sep=r'::', header=None, engine='python')
    for _, row in movie_df.iterrows():
        row_tmp = row.tolist()
        i_id = row_tmp[0]
        if i_id not in item_src2tgt_dict:   #item出现在item_cate列表中，但不在u-i-r列表中，应该去掉
            continue
        cate_list = row_tmp[2].strip().split('|')
        cate_list = [cate_src2tgt_dict[c] for c in cate_list]
        # item_feature_dict[i_id] = cate_list
        cate = np.random.choice(cate_list)
        genre_count[cate].append(i_id)
        # for cate in cate_list:
        #    genre_count[cate].append(i_id)
        item_cate_dict[item_src2tgt_dict[i_id]] = cate
    item_cate_dict = sorted(item_cate_dict.items(), key=lambda mydict: mydict[0], reverse=False)
    item_cate_dict = {i:j for i,j in item_cate_dict}
    cate_size = len(genre_count)
    print('user_num, item_num, cate_num, interaction_num:', user_size, item_size, cate_size, len(all_data))  # 6040 3706 18 1000209

    total_user_list = create_user_list(all_data, user_size)  # 把[u,i,r]转为list[u]={item...}
    print('avg. items per user: ', np.mean([len(u) for u in total_user_list]))  # 165.57


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
