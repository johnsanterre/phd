import numpy as np
import os
from collections import Counter

results = {}
for f_loc in ['k10.npy','k11.npy','k12.npy','k13.npy','k14.npy','k15.npy','k16.npy','k17.npy','k18.npy','k19.npy','k20.npy']:
  M = np.load(f_loc)
  results[f_loc]={}
  results[f_loc]['num_isolates']=M.shape[0]
  results[f_loc]['num_features']=M.shape[1]
  
  results[f_loc]['size_on_HD']=os.path.getsize(f_loc)
  
  unique, counts = np.unique(M, return_counts=True)
  cnts = dict(zip(unique, counts))
  results[f_loc]['counts']=cnts
  results[f_loc]['cnt_ones']=cnts[1]
  results[f_loc]['sparsity']=1-(cnts[0]/float(M.size))
  
  ret = np.vstack({tuple(row) for row in M.T}).T
  results[f_loc]['num_unique_feature_columns'] = ret.shape[1]
  print f_loc
  print results[f_loc]['num_isolates']
  print results[f_loc]['num_features']
  print results[f_loc]['size_on_HD']
  print results[f_loc]['cnt_ones']
  print results[f_loc]['sparsity']
  print "******"

np.save('results.npy', results) 

numFeat =  [results[f]['num_features'] for f in f_locs]
sparsity = [1-results[f]['sparsity'] for f in f_locs]
number_of_ones= [1-results[f]['cnt_ones'] for f in f_locs]
counts = [ [results[f]['counts'].keys(),  results[f]['counts'].values()] for f in f_locs]
unique_features = [ results[f]['num_unique_feature_columns'] for f in f_locs]
size = [ results[f]['size_on_HD'] for f in f_locs]
