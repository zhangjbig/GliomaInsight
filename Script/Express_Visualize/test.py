import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import make_interp_spline
import pandas as pd
from scipy.stats import lognorm
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import torch
import sys
sys.path.append("..")
from Final_version_dl_4 import FeatureTransformer, PositionalEncoding, ResidualBlock, DeepRadiomicsClassifier


class PredictAndPlot:
    def __init__(self):
        pass

    # 调用模型预测
    def prediction(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        best_model_path = os.path.join(current_dir, "dl_model/best_model_CCRL2.pth")
        model = FeatureTransformer(input_dim=14, d_model=32, nhead=4,
                                   nhid=512, nlayers=3, dropout=0.1)  #### 这里的输入维度要和训练时一致
        try:
            model.load_state_dict(torch.load(best_model_path))
        except Exception as e:
            print(f"模型加载出错: {e}")
            return None, None
        model.eval()

        data_path = os.path.join(current_dir, "test.csv")
        df = pd.read_csv(data_path)
        # print(f"数据基本信息: {df.info()}")
        # print(f"数据列数: {len(df.columns)}")
        scaler = StandardScaler()
        input_data = torch.tensor(scaler.fit_transform(df.iloc[:, 4:18].values), dtype=torch.float32)

        with torch.no_grad():
            probabilities = model(input_data)
            predictions = (probabilities > 0.5).int()

        pred_result = predictions.tolist()
        probability = probabilities.tolist()

        # print("Predicted Labels:", pred_result)
        # print("Probabilities:", probability)

        return predictions, probabilities

    # 初始化数据
    def plot_dynamic_curve(self, x, y, colname, new_x=None, new_y=None):
        fig, ax = plt.subplots()
        fig.patch.set_facecolor((15 / 255, 29 / 255, 43 / 255))
        ax.set_facecolor((15 / 255, 29 / 255, 43 / 255))

        ax.spines['bottom'].set_color((141 / 255, 185 / 255, 252 / 255))
        ax.spines['left'].set_color((141 / 255, 185 / 255, 252 / 255))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.tick_params(colors=(141 / 255, 185 / 255, 252 / 255))
        ax.set_xlabel("Gene Expression Level", fontsize=12, color=(141 / 255, 185 / 255, 252 / 255))
        ax.set_ylabel(f"Probability of {colname} Expression Level", fontsize=12, color=(141 / 255, 185 / 255, 252 / 255))

        line, = ax.plot([], [], color=(141 / 255, 185 / 255, 252 / 255), lw=2)

        def init():
            ax.set_xlim(min(x), max(x))
            ax.set_ylim(min(y), max(y))
            line.set_data([], [])
            return line,

        def update(frame):
            line.set_data(x[:frame], y[:frame])
            return line,

        ani = animation.FuncAnimation(fig, update, frames=len(x) + 1, init_func=init, interval=20, repeat=False)

        # 处理额外曲线
        if new_x is not None and new_y is not None:
            new_x = np.atleast_1d(new_x)
            new_y = np.atleast_1d(new_y)

            sorted_indices = np.argsort(new_x)
            new_x_sorted = new_x[sorted_indices]
            new_y_sorted = new_y[sorted_indices]

            if len(new_x_sorted) > 1:  # 只有多个点时才进行插值
                # 生成扰动数据
                perturbation = np.random.normal(scale=0.05 * np.max(new_y_sorted), size=len(new_y_sorted))
                new_y_perturbed = new_y_sorted + perturbation
                new_y_perturbed = np.clip(new_y_perturbed, 0, None)  # 保证概率不为负

                # 平滑处理
                spline = make_interp_spline(new_x_sorted, new_y_perturbed, k=3)
                smooth_x = np.linspace(min(new_x_sorted), max(new_x_sorted), 300)
                smooth_y = spline(smooth_x)

                ax.plot(smooth_x, smooth_y, color="#D0D7E1", lw=2)
            else:
                ax.scatter(new_x_sorted, new_y_sorted, color="#D0D7E1", marker="o", label="Perturbed Point")


        plt.show()

    def main(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data = pd.read_csv(os.path.join(current_dir, "TPM_CGGA.csv"))
        LogNorm = ['TRAM2']
        Gamma = ['CD40', 'KYNU', 'CCRL2', 'FCGR2A', 'JUNB', 'MBOAT1']
        for col_name in Gamma:
            print(f"正在处理列: {col_name}")
            # 拟合 Gamma 分布
            data_column = data[col_name]
            shape, loc, scale = stats.gamma.fit(data_column)
            if(col_name == 'CD40'):
                shape = 3.2443377626673273
                scale = 794.5733995413987
            x = np.linspace(min(data_column), max(data_column), 100)
            y = stats.gamma.pdf(x, shape, loc, scale)

            # 进行预测
            predictions, probabilities = self.prediction()
            if predictions is None or probabilities is None:
                continue

            # 数据转化
            max_level = max(data_column)
            min_level = min(data_column)
            x_new = probabilities*(max_level-min_level)*1000+min_level
            y_new = stats.gamma.pdf(x_new, shape, loc, scale)

            print(x_new.item(),y_new)
            self.plot_dynamic_curve(x, y, col_name,x_new.item(),y_new)
            # break

        # for col_name in LogNorm:
        #     print(f"正在处理列: {col_name}")
        #     # 拟合对数正态分布
        #     data_column = data[col_name]
        #     shape, loc, scale = lognorm.fit(data_column)
            
        #     x = np.linspace(min(data_column), 200, 100)
        #     print(max(data_column))
        #     y = lognorm.pdf(x, shape, loc, scale)

        #     self.plot_dynamic_curve(x, y,col_name)
        #     # plot_dynamic_curve(x, y, new_x, new_y)


if __name__ == '__main__':
    a = PredictAndPlot()
    a.main()