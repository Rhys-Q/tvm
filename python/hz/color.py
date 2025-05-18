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
        # self.k1_concentration = nn.Tensor.from_const(k1_concentration)           # [4, 1]
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

        # concentration
        self.ind_rank0 = nn.Tensor.from_const(np.take(alldata_concentration, rank_index[0], axis=0))
        self.ind_rank1 = nn.Tensor.from_const(np.take(alldata_concentration, rank_index[1], axis=0))
        self.ind_rank2 = nn.Tensor.from_const(np.take(alldata_concentration, rank_index[2], axis=0))
        self.ind_rank3 = nn.Tensor.from_const(np.take(alldata_concentration, rank_index[3], axis=0))

        # reflect
        def take_reflection(data, rank):
            # data: [4, IND_DIM, 31]
            # rank: [31]
            return data[rank[:, np.newaxis], np.arange(data.shape[1]), np.arange(31)[:, np.newaxis]]
        self.reflect_rank0 = nn.Tensor.from_const(take_reflection(alldata_reflection, rank_index[0]))
        self.reflect_rank1 = nn.Tensor.from_const(take_reflection(alldata_reflection, rank_index[1]))
        self.reflect_rank2 = nn.Tensor.from_const(take_reflection(alldata_reflection, rank_index[2]))
        self.reflect_rank3 = nn.Tensor.from_const(take_reflection(alldata_reflection, rank_index[3]))

        # K1
        def extract_k1_reflection(k1_reflection, rank):
            k1_rank = np.zeros([31], dtype=np.float32)
            for i in range(31):
                k1_rank[i] = k1_reflection[rank[i], 0, i]
            return k1_rank

        self.k1_rank0 = nn.Tensor.from_const(extract_k1_reflection(k1_reflection, rank_index[0]))
        self.k1_rank1 = nn.Tensor.from_const(extract_k1_reflection(k1_reflection, rank_index[1]))
        self.k1_rank2 = nn.Tensor.from_const(extract_k1_reflection(k1_reflection, rank_index[2]))
        self.k1_rank3 = nn.Tensor.from_const(extract_k1_reflection(k1_reflection, rank_index[3]))

        # W
        self.w_concentration_rank0 = nn.Tensor.from_const(np.take(w_concentration, rank_index[0], axis=0))
        self.w_concentration_rank1 = nn.Tensor.from_const(np.take(w_concentration, rank_index[1], axis=0))
        self.w_concentration_rank2 = nn.Tensor.from_const(np.take(w_concentration, rank_index[2], axis=0))
        self.w_concentration_rank3 = nn.Tensor.from_const(np.take(w_concentration, rank_index[3], axis=0))

        self.w_reflection_rank0 = nn.Tensor.from_const(take_reflection(w_reflection, rank_index[0]))
        self.w_reflection_rank1 = nn.Tensor.from_const(take_reflection(w_reflection, rank_index[1]))
        self.w_reflection_rank2 = nn.Tensor.from_const(take_reflection(w_reflection, rank_index[2]))
        self.w_reflection_rank3 = nn.Tensor.from_const(take_reflection(w_reflection, rank_index[3]))


    
    
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
        # IND_DIM is the num of ind, which is 1090 in this case
        sum_vars = nn.op.sum(vars, axis=1, keepdims=True) # [B, 1]
        vars = vars / sum_vars * 100
        # the first step
        vars0 = nn.op.take(vars, self.rank0, axis=1)
        vars1 = nn.op.take(vars, self.rank1, axis=1)
        vars2 = nn.op.take(vars, self.rank2, axis=1)
        vars3 = nn.op.take(vars, self.rank3, axis=1)

        # W
        c1 = vars1 / (vars1 + vars0) # [B, 31]

        def cal_index_in_ind(ind_rank, c):
            # ind_rank0: [31, IND_DIM]
            # c: [B, 31]
            ind = nn.op.unsqueeze(ind_rank, dim=0) # [1, 31, IND_DIM]
            c = nn.op.unsqueeze(c, dim=2) # [B, 31, 1]
            tmp = c <= ind
            ind = nn.op.argmax(tmp, axis=2).astype("int32") # [B, 31]
            return ind

        
        ind_index_c1 = cal_index_in_ind(self.ind_rank1, c1) # [B, 31]

        def cal_line(reflict_1, ind_1,c1, ind_c1, i, j):
            # 根据浓度算曲线
            # reflict_1: [31, IND_DIM]
            # ind_1: [31, IND_DIM]
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
            return _tir.Select( ind_index == 0, reflict_1[j, 0], 
                            cal_line_intern(reflict_1[j,ind_index],reflict_1[j,ind_index-1], ind_1[j, ind_index],ind_1[j, ind_index-1],c1[i,j] )
                    )

        def cal_concentration(reflict, ind , r, ind_r, i, j):
            # reflict: [31, IND_DIM]
            # ind: [31, IND_DIM]
            # r: [B, 31]
            # ind_r: [B, 31]
            # i [0, B)
            # j [0, 31)
            def cal_concentration_intern(ind1, ind2, v1, v2, w):
                return ind1 - (ind1 - ind2)/(v1-v2) * (v1 - w)
            
            ind_index = ind_r[i, j]
            return _tir.Select( ind_index == 0, ind[j, 0], 
                cal_concentration_intern(ind[j, ind_index-1], ind[j, ind_index], reflict[j,ind_index-1], reflict[j,ind_index], r[i,j])
            )

        r1 = nn.op.tensor_expr_op(
            lambda reflict_1, ind_1,c1, ind_c1: te.compute(
                ind_c1.shape,
                lambda i, j:  cal_line(reflict_1, ind_1,c1, ind_c1, i, j),
                name="cal_line",
            ),
            "cal_line",
            args=[self.reflect_rank1, self.ind_rank1,c1, ind_index_c1],
        ) # [B, 31]

        def cal_index_in_reflection(reflict, r):
            # reflect: [31, IND_DIM]
            # r: [B, 31]
            tmp = nn.unsqueeze(r, dim=2) >= nn.unsqueeze(reflict, dim=0)
            index = nn.op.argmax(tmp, axis=2).astype("int32") # [B, 31]
            return index
        index_r1 = cal_index_in_reflection(self.reflect_rank2, r1) # [B, 31]
        M3 = nn.op.tensor_expr_op(
            lambda reflict, ind,r, ind_r: te.compute(
                ind_r.shape,
                lambda i, j:  cal_concentration(reflict, ind,r, ind_r, i, j),
                name="cal_concentration",
            ),
            "cal_concentration",
            args=[self.reflect_rank2, self.ind_rank2,r1, index_r1],
        ) # [B, 31]

        # C2
        def cal_C2():
            KValue = self.k1_rank2 # [31]
            KR2And3 = KValue * vars2 / (KValue *vars2 + vars1) # [B, 31]
            index_w = cal_index_in_ind(self.w_concentration_rank1, KR2And3) #  [B, 31]
            w2 = nn.op.tensor_expr_op(
                lambda reflict_1, ind_1,c1, ind_c1: te.compute(
                    ind_c1.shape,
                    lambda i, j:  cal_line(reflict_1, ind_1,c1, ind_c1, i, j),
                    name="cal_line",
                ),
                "cal_line",
                args=[self.w_reflection_rank1, self.w_concentration_rank1,KR2And3, index_w],
            ) # [B, 31]
            C2_1 = ((vars0+ vars1 * w2)/(1- M3 ) - vars0 - vars1* w2 + vars2)
            C2_2 = (vars0 + vars1 * w2) / (1 - M3 + vars2)
            C2 = C2_1 / C2_2
            return C2

        C2 = cal_C2()

        # 计算R2
        index_c2 = cal_index_in_ind(self.ind_rank2, C2) #[B, 31]
        R2 = nn.op.tensor_expr_op(
            lambda reflict, ind,c, index: te.compute(
                index.shape,
                lambda i, j:  cal_line(reflict, ind,c, index, i, j),
                name="cal_line",
            ),
            "cal_line",
            args=[self.reflect_rank2, self.ind_rank2,C2, index_c2],
        ) # [B, 31]

        # 计算M4
        index_R2 = cal_index_in_reflection(self.reflect_rank3, R2) # [B, 31]
        M4 = nn.op.tensor_expr_op(
            lambda reflict, ind,r, ind_r: te.compute(
                ind_r.shape,
                lambda i, j:  cal_concentration(reflict, ind,r, ind_r, i, j),
                name="cal_concentration",
            ),
            "cal_concentration",
            args=[self.reflect_rank3, self.ind_rank3,R2, index_R2],
        ) # [B, 31]

        def cal_C3():
            KValue = self.k1_rank3 # [31]
            KR2And4 = KValue* vars3 / (vars1+ vars3 * KValue)# [B, 31]
            KR3And4 = KValue * vars3 / (KValue *vars3 + vars2) # [B, 31]
            index_w2_4 = cal_index_in_ind(self.w_concentration_rank1, KR2And4) #  [B, 31]
            index_w3_4 = cal_index_in_ind(self.w_concentration_rank2, KR3And4) #  [B, 31]
            w2_4 = nn.op.tensor_expr_op(
                lambda reflict_1, ind_1,c1, ind_c1: te.compute(
                    ind_c1.shape,
                    lambda i, j:  cal_line(reflict_1, ind_1,c1, ind_c1, i, j),
                    name="cal_line",
                ),
                "cal_line",
                args=[self.w_reflection_rank1, self.w_concentration_rank1,KR2And4, index_w2_4],
            ) # [B, 31]
            w3_4 = nn.op.tensor_expr_op(
                lambda reflict_1, ind_1,c1, ind_c1: te.compute(
                    ind_c1.shape,
                    lambda i, j:  cal_line(reflict_1, ind_1,c1, ind_c1, i, j),
                    name="cal_line",
                ),
                "cal_line",
                args=[self.w_reflection_rank2, self.w_concentration_rank2,KR3And4, index_w3_4],
            ) # [B, 31]

            C3_1 =((vars0 + vars1* w2_4 + vars2* w3_4) / (1 - M4) - vars0 - vars1* w2_4 - vars2* w3_4 + vars3)
            C3_2 = ((vars0 + vars1* w2_4 + vars2* w3_4)/(1-M4) +  vars3)
            C3 = (C3_1 / C3_2)
            return C3
        C3 = cal_C3()

        # R3
        index_c3 = cal_index_in_ind(self.ind_rank3, C3) #[B, 31]
        R3 = nn.op.tensor_expr_op(
            lambda reflict, ind,c, index: te.compute(
                index.shape,
                lambda i, j:  cal_line(reflict, ind,c, index, i, j),
                name="cal_line",
            ),
            "cal_line",
            args=[self.reflect_rank3, self.ind_rank3,C3, index_c3],
        ) # [B, 31]
        return R3





if __name__ == "__main__":
    config = ColorConfig()

    cm = ColorMatch(config)
    forward_spec = {"color_match": {"vars": spec.Tensor([10, 4], dtype="float32")}}
    cm.jit(forward_spec, debug=True)