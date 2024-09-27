import pickle
from scipy.io import loadmat
import numpy as np
import os


X_list = []
y_list = []

for i in range(123):
    file_name = f'sub{i:03d}.pkl.pkl'
    file_path_data = os.path.join('EEG_Features/DE', file_name)
    file_path_mat = f'Data/sub{i:03d}/After_remarks.mat'
    with open(file_path_data, 'rb') as f:
        data = pickle.load(f)

    if i <61:
        indices_to_delete = [15, 16, 17, 18, 26, 27]
        #indices_to_delete = [11, 14, 15, 16, 17, 25, 26]
        data = np.delete(data, indices_to_delete, axis=1)
    else:
        indices_to_delete = [11, 14, 15, 16, 17, 23, 24, 25, 26, 30, 31]
        indices_to_delete = [11, 14, 15, 23, 24, 30, 31]
        data = np.delete(data, indices_to_delete, axis=1)

    mat_data = loadmat(file_path_mat)
    variable_name = 'After_remark'
    my_variable_value = mat_data[variable_name]

    for j in range(len(data)):
        X_list.append(data[j])
        y_list.append(my_variable_value[j][0][0][0])

X = np.array(X_list)
y = np.array(y_list)


with open('DEX_data.pkl', 'wb') as f:
    pickle.dump(X, f)


with open('DEy_data.pkl', 'wb') as f:
    pickle.dump(y, f)
