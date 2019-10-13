# -*- coding: utf-8 -*-
"""
create on Sep 24, 2019

@author: wangshuo
"""

import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat

random.seed(1234)

workdir = 'datasets/'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Ciao', help='dataset name: Ciao/Epinions')
parser.add_argument('--test_prop', default=0.1, help='the proportion of data used for test')
args = parser.parse_args()

# load data
if args.dataset == 'Ciao':
	click_f = loadmat(workdir + 'Ciao/rating.mat')['rating']
	trust_f = loadmat(workdir + 'Ciao/trustnetwork.mat')['trustnetwork']
elif args.dataset == 'Epinions':
	click_f = np.loadtxt(workdir+'Epinions/ratings_data.txt', dtype = np.int32)
	trust_f = np.loadtxt(workdir+'Epinions/trust_data.txt', dtype = np.int32)
else:
	pass 

click_list = []
trust_list = []

u_items_list = []
u_users_list = []
u_users_items_list = []
i_users_list = []

user_count = 0
item_count = 0
rate_count = 0

for s in click_f:
	uid = s[0]
	iid = s[1]
	if args.dataset == 'Ciao':
		label = s[3]
	elif args.dataset == 'Epinions':
		label = s[2]

	if uid > user_count:
		user_count = uid
	if iid > item_count:
		item_count = iid
	if label > rate_count:
		rate_count = label
	click_list.append([uid, iid, label])

pos_list = []
for i in range(len(click_list)):
	pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2]))

# remove duplicate items in pos_list because there are some cases where a user may have different rate scores on the same item.
pos_list = list(set(pos_list))

# train, valid and test data split
random.shuffle(pos_list)
num_test = int(len(pos_list) * args.test_prop)
test_set = pos_list[:num_test]
valid_set = pos_list[num_test:2 * num_test]
train_set = pos_list[2 * num_test:]
print('Train samples: {}, Valid samples: {}, Test samples: {}'.format(len(train_set), len(valid_set), len(test_set)))

with open(workdir + args.dataset + '/dataset.pkl', 'wb') as f:
	pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)


train_df = pd.DataFrame(train_set, columns = ['uid', 'iid', 'label'])
valid_df = pd.DataFrame(valid_set, columns = ['uid', 'iid', 'label'])
test_df = pd.DataFrame(test_set, columns = ['uid', 'iid', 'label'])

click_df = pd.DataFrame(click_list, columns = ['uid', 'iid', 'label'])
train_df = train_df.sort_values(axis = 0, ascending = True, by = 'uid')

"""
u_items_list: 存储每个用户交互过的物品iid和对应的评分，没有则为[(0, 0)]
"""
for u in tqdm(range(user_count + 1)):
	hist = train_df[train_df['uid'] == u]
	u_items = hist['iid'].tolist()
	u_ratings = hist['label'].tolist()
	if u_items == []:
		u_items_list.append([(0, 0)])
	else:
		u_items_list.append([(iid, rating) for iid, rating in zip(u_items, u_ratings)])

train_df = train_df.sort_values(axis = 0, ascending = True, by = 'iid')

"""
i_users_list: 存储与每个物品相关联的用户及其评分，没有则为[(0, 0)]
"""
for i in tqdm(range(item_count + 1)):
	hist = train_df[train_df['iid'] == i]
	i_users = hist['uid'].tolist()
	i_ratings = hist['label'].tolist()
	if i_users == []:
		i_users_list.append([(0, 0)])
	else:
		i_users_list.append([(uid, rating) for uid, rating in zip(i_users, i_ratings)])

for s in trust_f:
	uid = s[0]
	fid = s[1]
	if uid > user_count or fid > user_count:
		continue
	trust_list.append([uid, fid])

trust_df = pd.DataFrame(trust_list, columns = ['uid', 'fid'])
trust_df = trust_df.sort_values(axis = 0, ascending = True, by = 'uid')


"""
u_users_list: 存储每个用户互动过的用户uid；
u_users_items_list: 存储用户每个朋友的物品iid列表
"""
for u in tqdm(range(user_count + 1)):
	hist = trust_df[trust_df['uid'] == u]
	u_users = hist['fid'].unique().tolist()
	if u_users == []:
		u_users_list.append([0])
		u_users_items_list.append([[(0,0)]])
	else:
		u_users_list.append(u_users)
		uu_items = []
		for uid in u_users:
			uu_items.append(u_items_list[uid])
		u_users_items_list.append(uu_items)
	
with open(workdir + args.dataset + '/list.pkl', 'wb') as f:
	pickle.dump(u_items_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(u_users_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(u_users_items_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(i_users_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump((user_count, item_count, rate_count), f, pickle.HIGHEST_PROTOCOL)


