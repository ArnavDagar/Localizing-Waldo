import pickle

matrix_file = 'matrix.p'

with open(matrix_file, mode='rb') as f:
    data = pickle.load(f)
