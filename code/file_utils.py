import os.path

def make_file_names(name, directory='./'):
  ret = []
  for i in range(10,21):
    file_name = directory + name + str(i) + '.npy'
    if os.path.exists(file_name):
      ret.append(file_name)
  return ret
