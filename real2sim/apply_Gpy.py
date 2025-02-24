import GPy
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score


def obtain_model():
    # read column with 0, 1 from csv file as numpy array
    real_data = pd.read_csv('./data/Gpy_data/real_data.csv', header=None).to_numpy()[1:, 1:].reshape(-1, 2)
    sim_data = pd.read_csv('./data/Gpy_data/sim_data.csv', header=None).to_numpy()[1:, 1:].reshape(-1, 2)  

    # Create a GPy model
    kernel = GPy.kern.ThinPlate(input_dim=2, variance=3., R=1000)
    model_sx = GPy.models.GPRegression(real_data,sim_data[:, 0].reshape(-1,1),kernel)    # sensor model in x direction
    model_sy = GPy.models.GPRegression(real_data,sim_data[:, 1].reshape(-1,1),kernel)    # sensor model in y direction

    # save those two models
    return model_sx, model_sy

def apply_GPy():
    # load the model
    model_sx, model_sy = obtain_model()
    file_name_list = [file_name.split('.')[0] for file_name in os.listdir('./data/real_world_data')]
    file_dict = {}
    for f in file_name_list:
        file_words = f.split('_')
        for i, s in enumerate(file_words):
            if s == 'action':
                file_name = '_'.join(file_words[:i])
                if file_name not in file_dict.keys():
                    file_dict[file_name] = {} 
                file_dict[file_name]["stage"] = f
            elif s == 'optic':
                file_name = '_'.join(file_words[:i])
                if file_name not in file_dict.keys():
                    file_dict[file_name] = {} 
                file_dict[file_name]["optic"] = f

    for k in file_dict.keys(): 
        object_name = k
        stage = pd.read_csv(f'data/real_world_data/{file_dict[object_name]["stage"]}.csv')
        optic = pd.read_csv(f'data/real_world_data/{file_dict[object_name]["optic"]}.csv')

        stage = np.array(stage[['py','ind']])
        temp = np.array(optic[['fbg_4']], dtype=np.float32)
        optic = np.array(optic[['fbg_2', 'fbg_6']], dtype=np.float32)
        tmp = deepcopy(temp)
        for i in range(300, len(temp)):
            temp[i] = np.mean(tmp[i-300:i])
        # fill the first 100 elements with the mean of the first 100 elements
        temp[:300] = np.mean(temp[:300])
        optic = optic - temp

        optic = optic[int(stage[0, 1]):int(stage[-1, 1])+1, :]
        init_sig  = optic[1500:1700]
        
        init_sig = np.mean(init_sig, axis=0)
        optic = optic - init_sig
        max_optic = 0.12
        min_optic = - 0.37
        optic = (optic - min_optic) / (max_optic - min_optic)
        
        mapped_sim_x = []
        mapped_sim_y = []
        for i in range(optic.shape[0]):
            mapped_sim_x.append(model_sx.predict(optic[i].reshape(1, -1))[0][0][0])
            mapped_sim_y.append(model_sy.predict(optic[i].reshape(1, -1))[0][0][0])
        mapped_sim_x = np.array(mapped_sim_x)
        mapped_sim_y = np.array(mapped_sim_y)

        mapped_sim = np.array([mapped_sim_x, mapped_sim_y]).T

        mapped_sim = mapped_sim - mapped_sim[0, :] + np.array([40000/(40000 + 30000), 40000/(40000 + 30000)])
        mapped_sim = mapped_sim[:, :2] * (40000 + 30000) - 40000

        stage = stage[:, 0]
        stage = np.interp(np.linspace(0, 1, optic.shape[0]), np.linspace(0, 1, stage.shape[0]), stage)
        mapped_optic_df = pd.DataFrame({'sx': mapped_sim[:, 0], 'sy': mapped_sim[:, 1], 'px': stage})
        mapped_optic_df.to_csv(f'../whiskerNet/data/test/{object_name}_mapped_sim.csv', index=False)

if __name__ == "__main__":
    apply_GPy()