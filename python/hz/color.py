import os

from tvm import relax, te
from tvm.relax.frontend import nn
import numpy as np
from tvm.relax.frontend.nn import spec
from tvm import tir as _tir
class ColorConfig():
    def __init__(self, data_path = "/home/hz/qzq_work/color_train_dataset/data/BaseData_3009", systemid=1, matnumbers: list[int]= [0, 1, 12,21]):
        self.systemid = systemid
        self.data_path = data_path
        self.matnumbers = matnumbers

class ColorMatch(nn.Module):
    def __init__(self, config: ColorConfig):
        super().__init__()

        self.config = config

        alldata_concentration, alldata_reflection, k1_concentration, k1_reflection, w_concentration, w_reflection = self._load_data()

        self.alldata_concentration = nn.Tensor.from_const(alldata_concentration) # [4, 1090]
        self.alldata_reflection = nn.Tensor.from_const(alldata_reflection)       # [4, 1090, 31]
        self.k1_concentration = nn.Tensor.from_const(k1_concentration)           # [4, 1]
        self.k1_reflection = nn.Tensor.from_const(k1_reflection)                 # [4, 1, 31]
        self.w_concentration = nn.Tensor.from_const(w_concentration)             # [4, 658]
        self.w_reflection = nn.Tensor.from_const(w_reflection)                   # [4, 658, 31]

        last_alldata = alldata_reflection[:,-1,:]
        rank_index = np.argsort(-last_alldata, axis=0).astype(np.int32)
        # self.rank_index = nn.Tensor.from_const(rank_index) # [4, 31]
        self.rank0 = nn.Tensor.from_const(rank_index[0])
        self.rank1 = nn.Tensor.from_const(rank_index[1])
        self.rank2 = nn.Tensor.from_const(rank_index[2])
        self.rank3 = nn.Tensor.from_const(rank_index[3])

    
    
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

        return alldata_concentration, alldata_reflection, k1_concentration, k1_reflection, w_concentration, w_reflection
    
    
    
    def color_match(self, vars: nn.Tensor):
        # https://gitee.com/qzqbiubiubiu/cal_color/blob/master/src/cal_color/color4.cc
        # vars shape is [B, 4]
        sum_vars = nn.op.sum(vars, axis=1, keepdims=True) # [B, 1]
        vars = vars / sum_vars * 100
        # the first step
        tmp1 = nn.op.take(vars, self.rank0, axis=1)
        tmp2 = nn.op.take(vars, self.rank1, axis=1)
        c1 = tmp2 / (tmp1 + tmp2) # [B, 31]


        ind_0 = nn.op.take(self.alldata_concentration, self.rank1, axis=0) # [31, IND_DIM]
        ind = nn.op.unsqueeze(ind_0, dim=0) # [1, 31, IND_DIM]
        c1_0 = nn.op.unsqueeze(c1, dim=2) # [B, 31, 1]
        tmp3 = c1_0 > ind
        ind_c1 = nn.op.argmax(tmp3, axis=2).astype("int32") # [B, 31]

        def take_reflection(data, rank):
            t = nn.op.tensor_expr_op(
                lambda data, rank: te.compute(
                    [rank.shape[0], data.shape[1]],
                    lambda i, j:  data[rank[i], j, i],
                    name="cal_line",
                ),
                "cal_line",
                args=[data, rank],
                ) # [B, 31]
            return t
        reflict_1 = take_reflection(self.alldata_reflection, self.rank1)# [31, REF_DIM]
        def cal_line(reflict_1, ind_1,c1, ind_c1, i, j):
            # reflict_1: [31, REF_DIM]
            # ind_1: [31, REF_DIM]
            # c1: [B, 31]
            # ind_c1: [B, 31]
            # i [0, B)
            # j [0, 31)
            def ind_convert(ind):
                return ind / (1-ind)
            def cal_line_intern(v1, v2, ind1, ind2, w):
                ind1 = ind_convert(ind1)
                ind2 = ind_convert(ind2)
                w = ind_convert(w)
                return v1 - (v1-v2)/(ind1 - ind2) * (ind1 - w)
            
            ind_index = ind_c1[i, j]
            num = reflict_1.shape[1]
            return _tir.Select( ind_index == 0, _tir.const(0, "float32"), 
                        _tir.Select(ind_index == num-1,reflict_1[j, num-1], 
                            cal_line_intern(reflict_1[j,ind_index],reflict_1[j,ind_index-1],    ind_1[j, ind_index],ind_1[j, ind_index-1],c1[i,j] )
                        )
                    )
            

        r1 = nn.op.tensor_expr_op(
            lambda reflict_1, ind_1,c1, ind_c1: te.compute(
                ind_c1.shape,
                lambda i, j:  cal_line(reflict_1, ind_1,c1, ind_c1, i, j),
                name="cal_line",
            ),
            "cal_line",
            args=[reflict_1, ind_0,c1, ind_c1],
        ) # [B, 31]

        reflict_2 = take_reflection(self.alldata_reflection, self.rank2) # [31, REF_DIM]
        tmp = nn.unsqueeze(r1, dim=2) > nn.unsqueeze(reflict_2, dim=0)
        index_r1 = nn.op.argmax(tmp, axis=2).astype("int32") # [B, 31]




        

        return r1






if __name__ == "__main__":
    config = ColorConfig()

    cm = ColorMatch(config)
    forward_spec = {"color_match": {"vars": spec.Tensor([10, 4], dtype="float32")}}
    cm.jit(forward_spec, debug=True)