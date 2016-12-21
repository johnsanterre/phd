#python kmerge analoge

import numpy as np

f_loc= 'Mycobacterium_ethambutol.log'
for kmer_size in [10,11,12]:
  source_loc = '/homes/jsanterre/data/raw/amr/Mycobacterium/ethambutol/counts/All/k'+str(kmer_size)+'/'
  np_M_name = 'k'+str(kmer_size)+'.npy'
  with open(f_loc, 'r+') as handle:
    files = []
    for x in handle:
      loc,t = x.split()
      loc = source_loc+loc
      files.append([loc,t])
  kmers = {}
  for row_index, rec in enumerate(files):
    fname, phenotype = rec
    with open(fname+'.k'+str(kmer_size), 'r') as f:
        for line in f:
            k,v = line.split()
            if k not in kmers:
              kmers[k] = {row_index:int(v)}
            else:
              kmers[k][row_index] = int(v)
  number_kmers = len(kmers.keys())
  M = np.zeros((row_index+1, number_kmers), dtype='uint16')
  for col_id, kmer_string in enumerate(kmers.keys()):
    for row_id in kmers[kmer_string].keys():
      M[row_id, col_id] = kmers[kmer_string][row_id]
  np.save(np_M_name,M)
