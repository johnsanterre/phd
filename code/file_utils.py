import os.path
import csv

def get_labels(name, directory = '../../'):
  ret = []
  file_name = directory + name
  if os.path.exists(file_name):
    with open(file_name, 'rb') as csvfile:
       data = csv.reader(csvfile, delimiter=' ')
       ret = [d for d in data]
    return ret

def make_file_names(name, directory='../../'):
  ret = []
  for i in range(10,21):
    file_name = directory + name +'_k' + str(i) + '.npy'
    if os.path.exists(file_name):
      ret.append(file_name)
  return ret
