import os

from tvm import relax
from tvm.relax.frontend import nn
import numpy as np

class ColorConfig():
    def __init__(self, data_path = "/home/hz/qzq_work/color_train_dataset/data/BaseData_3009", systemid=1, matnumbers: list[int]= [0, 1, 12,21]):
        self.systemid = systemid
        self.data_path = data_path
        self.matnumbers = matnumbers

class ColorMatch():
    def __init__(self, config: ColorConfig):
        super().__init__()

        self.config = config

        self._load_data()
    
    def _load_data(self):
        def _load_std_data(data_path, num_std):
            with open(data_path, "r") as f:
                data = f.readlines()
                num = int(data[0].rstrip())
                concentration = [float(d) for d in data[1].rstrip().split()]
                concentration = np.array(concentration, dtype=np.float32).reshape(num)

                lines = []
                for i in range(2, 2+num):
                    line = [float(d) for d in data[i].rstrip().split()]
                    lines.append(line)
                lines = np.array(lines, dtype=np.float32).reshape(num, 31)
                assert lines.shape == (num, 31)
            
            assert num == 1 or num == num_std

            if num == 1:
                concentration = np.repeat(concentration, num_std)
                lines = np.repeat(lines, num_std, axis=0)            
            return concentration, lines
        
        def _get_num(data_path):
            with open(data_path, "r") as f:
                data = f.readlines()
                num = int(data[0].rstrip())                
                return num

        # alldata
        base_path = os.path.join(self.config.data_path, f"{self.config.systemid}")
        num_std = _get_num(os.path.join(base_path, "ALLDATA", "1.txt"))
        alldata_concentration, alldata_reflection = [],[]
        for mat in self.config.matnumbers:
            data_path = os.path.join(base_path, "ALLDATA", f"{mat}.txt")
            concentration, lines = _load_std_data(data_path, num_std)
            alldata_concentration.append(concentration)
            alldata_reflection.append(lines)
        alldata_concentration = np.stack(alldata_concentration, axis=0)
        alldata_reflection = np.stack(alldata_reflection, axis=0)
        self.alldata_reflection = alldata_reflection
        self.alldata_concentration = alldata_concentration

        # K1
        k1_concentration, k1_reflection = [],[]
        num_std = _get_num(os.path.join(base_path, "K1", "K1_1.txt"))
        for mat in self.config.matnumbers:
            data_path = os.path.join(base_path, "K1", f"K1_{mat}.txt")
            concentration, lines = _load_std_data(data_path, num_std)
            k1_concentration.append(concentration)
            k1_reflection.append(lines)
        k1_concentration = np.stack(k1_concentration, axis=0)
        k1_reflection = np.stack(k1_reflection, axis=0)

        self.k1_reflection = k1_reflection
        self.k1_concentration = k1_concentration

        # W
        w_concentration, w_reflection = [],[]
        num_std = _get_num(os.path.join(base_path, "W", "W_2.txt"))
        for mat in self.config.matnumbers:
            data_path = os.path.join(base_path, "W", f"W_{mat}.txt")
            concentration, lines = _load_std_data(data_path, num_std)
            w_concentration.append(concentration)
            w_reflection.append(lines)
        w_concentration = np.stack(w_concentration, axis=0)
        w_reflection = np.stack(w_reflection, axis=0)

        self.w_reflection = w_reflection
        self.w_concentration = w_concentration



if __name__ == "__main__":
    config = ColorConfig()

    cm = ColorMatch(config)
    breakpoint()