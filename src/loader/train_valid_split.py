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
      
      sum1 = (ann[:, :, 1] == 1).sum()
      sum2 = (ann[:, :, 1] == 2).sum()
      sum3 = (ann[:, :, 1] == 3).sum()
      sum4 = (ann[:, :, 1] == 4).sum()
      sum_all = sum1 + sum2 + sum3 + sum4
      
      ratios.append([sum1 / sum_all, sum2 / sum_all, sum3 / sum_all, sum4 / sum_all])
      files.append(arr_name)
      cnt += 1
      
      #if cnt > 100:
      #  break
    except:
      import traceback
      print(f' !!!! Failed to read input: {arr_name}')
      traceback.print_exc(file=sys.stdout)
      
  #hist_info = plt.hist(ratios, bins=10, density=True)  
  
  ratios_valid = []
  for arr_name in glob.glob(f'{valid_path}/*.npy'):
    
    print(arr_name)
    try:
    
      data = np.load(arr_name)
      
      img = data[...,:3] # RGB images
      ann = data[...,3:] # instance ID map
      
      
      sum1 = (ann[:, :, 1] == 1).sum()
      sum2 = (ann[:, :, 1] == 2).sum()
      sum3 = (ann[:, :, 1] == 3).sum()
      sum4 = (ann[:, :, 1] == 4).sum()
      sum_all = sum1 + sum2 + sum3 + sum4
      
      ratio = [sum1 / sum_all, sum2 / sum_all, sum3 / sum_all, sum4 / sum_all]
      ratios.append(ratio)
      ratios_valid.append(ratio)
      files.append(arr_name)
      cnt += 1
      
      print(ratio)
      
    except:
      import traceback
      print(f' !!!! Failed to read input: {arr_name}')
      traceback.print_exc(file=sys.stdout)
  
  bins = [0.01, 0.1, 0.5]
  bin_vals = np.digitize(ratios, bins=bins)
  fake_cls = bin_vals[:, 0] * 4**3 + bin_vals[:, 1] * 4**2 + bin_vals[:, 2] * 4 + bin_vals[:, 1]
  
  bin_valsv = np.digitize(ratios_valid, bins=bins)
  fake_clsv = bin_valsv[:, 0] * 4**3 + bin_valsv[:, 1] * 4**2 + bin_valsv[:, 2] * 4 + bin_valsv[:, 1]  
  
  
  plt.hist(fake_cls, bins=40, density=True) 
  plt.hist(fake_clsv, bins=40, density=True)   
  
  file_ids = np.arange(len(files))
  folds = 5
  fake_cls = np.asarray(fake_cls)
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
  
  bin_valsv2 = np.digitize(new_valid_ratios, bins=bins)
  fake_clsv2 = bin_valsv2[:, 0] * 4**3 + bin_valsv2[:, 1] * 4**2 + bin_valsv2[:, 2] * 4 + bin_valsv2[:, 1]    
  
  hist_info3 = plt.hist(fake_clsv2, bins=40, density=True, rwidth=0.25)  
  plt.show()  

  
  
      