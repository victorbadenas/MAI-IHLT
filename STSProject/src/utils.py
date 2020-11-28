import os

def read_file(file_path, as_list=True, delimiter=' '):
    with open(file_path, 'r') as f:
        data = f.readlines()
    if as_list:
        return list(map(lambda line: line.strip().split(delimiter), data))
    return data
