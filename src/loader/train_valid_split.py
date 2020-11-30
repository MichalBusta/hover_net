'''
Created on Nov 30, 2020

@author: michal.busta at gmail.com
'''

import glob, sys, os
import argparse

import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import shutil

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--dataroot', default='/home/busta/data/consep/CoNSeP/train/540x540_80x80', help='path to dataset')
  args = parser.parse_args() 
  
  ratios = []
  files = []
  cnt = 0
  valid_path = args.dataroot.replace('/train/', '/valid/')
  test_path = args.dataroot.replace('/train/', '/test/')
  for arr_name in glob.glob(f'{args.dataroot}/*.npy'):
    
    print(arr_name)
    try:
    
      data = np.load(arr_name)
      
      img = data[...,:3] # RGB images
      ann = data[...,3:] # instance ID map
      
      sum_cls0 = ann[:, :, 0].sum()
      sum_cls1 = ann[:, :, 1].sum()
      
      ratios.append(sum_cls1 / sum_cls0)
      files.append(arr_name)
      
      print(f'{sum_cls0}/{sum_cls1} - ratio: {sum_cls1 / sum_cls0}')
      cnt += 1
      
      if cnt > 100:
        break
    except:
      import traceback
      print(f' !!!! Failed to read input: {arr_name}')
      traceback.print_exc(file=sys.stdout)
      
  hist_info = plt.hist(ratios, bins=10, density=True)  
  
  ratios_valid = []
  for arr_name in glob.glob(f'{valid_path}/*.npy'):
    
    print(arr_name)
    try:
    
      data = np.load(arr_name)
      
      img = data[...,:3] # RGB images
      ann = data[...,3:] # instance ID map
      
      sum_cls0 = ann[:, :, 0].sum()
      sum_cls1 = ann[:, :, 1].sum()
      
      ratios_valid.append(sum_cls1 / sum_cls0)
      ratios.append(sum_cls1 / sum_cls0)
      files.append(arr_name)
      
      print(f'{sum_cls0}/{sum_cls1} - ratio: {sum_cls1 / sum_cls0}')
      cnt += 1
      
    except:
      import traceback
      print(f' !!!! Failed to read input: {arr_name}')
      traceback.print_exc(file=sys.stdout)
  
  
  hist_info2 = plt.hist(ratios_valid, bins=hist_info[1], density=True, rwidth=0.5)  
  fake_cls = []
  file_ids = []
  for fid, f in enumerate(files):
    y = []
    y.append(ratios[fid] < hist_info2[1][0])
    for bid in range(0, len(hist_info2[1]) - 1):
      y.append(ratios[fid] >= hist_info2[1][bid] and ratios[fid] < hist_info2[1][bid + 1])
    y.append(ratios[fid] >= hist_info2[1][-1])
    y = np.asarray(y).astype(np.int)
    fake_cls.append(y)
    file_ids.append(fid)
  
  folds = 5
  fake_cls = np.asarray(fake_cls)
  fake_cls = fake_cls.argmax(1)
  skf = StratifiedKFold(n_splits=folds, random_state=2)
  skf.get_n_splits(file_ids, fake_cls)
  
  for train_index, test_index in skf.split(file_ids, fake_cls):
    break
  
  new_train_path = args.dataroot.replace('/train/', '/train_strat/')
  if not os.path.exists(new_train_path):
    os.makedirs(new_train_path)
  for idx in train_index:
    dst = files[idx].replace('/train/', '/train_strat/')
    dst = dst.replace('/valid/', '/train_strat/')
    shutil.copy(files[idx], dst, follow_symlinks=True)
  
  new_valid_path = args.dataroot.replace('/train/', '/valid_strat/')
  if not os.path.exists(new_valid_path):
    os.makedirs(new_valid_path)  
  new_valid_ratios = []
  for idx in test_index:
    dst = files[idx].replace('/train/', '/valid_strat/')
    dst = dst.replace('/valid/', '/valid_strat/')
    shutil.copy(files[idx], dst, follow_symlinks=True)
    new_valid_ratios.append(ratios[idx])
    
  
  hist_info3 = plt.hist(new_valid_ratios, bins=hist_info[1], density=True, rwidth=0.25)  
  plt.show()  

  
  
      