# cd "D:/www/xxx/tumorTest/MTL/code_test/Visualize"
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from scipy.stats import lognorm
import torch
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.stats import cauchy


class PlotCanvas(FigureCanvas):
    def __init__(self, gene_name, prediction=1, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.gene_name = str(gene_name)
        self.prediction = int(prediction)

        # 设置主图样式
        self.fig.patch.set_facecolor((15 / 255, 29 / 255, 43 / 255))
        self.ax.set_facecolor((15 / 255, 29 / 255, 43 / 255))
        self.ax.spines['bottom'].set_color((141 / 255, 185 / 255, 252 / 255))
        self.ax.spines['left'].set_color((141 / 255, 185 / 255, 252 / 255))
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.tick_params(colors=(141 / 255, 185 / 255, 252 / 255))
        self.ax.set_xlabel(f"{self.gene_name} Expression Level", fontsize=12, color=(141 / 255, 185 / 255, 252 / 255))
        self.ax.set_ylabel(f"Probability Distribution of {self.gene_name}", fontsize=12, color=(141 / 255, 185 / 255, 252 / 255))

        # 设置右轴
        self.ax_right = self.ax.twinx()
        self.ax_right.spines['right'].set_color((255 / 255, 152 / 255, 0 / 255))
        self.ax_right.tick_params(colors=(255 / 255, 152 / 255, 0 / 255))
        self.ax_right.set_ylabel(f"Actual Probability Density of {self.gene_name}", fontsize=12, color=(255 / 255, 152 / 255, 0 / 255))

    def plot_static_curve(self, x, y, new_x=None, new_y=None, l_x=None, r_x=None):
        # 清空旧图但保留轴结构
        self.ax.cla()
        self.ax_right.remove()  # 删除旧的右轴（注意：不是 cla()）
        self.ax_right = self.ax.twinx()  # 重新建立右轴，绑定当前 ax

        # 设置主图样式
        self.ax.set_facecolor((15 / 255, 29 / 255, 43 / 255))
        self.ax.spines["bottom"].set_color((141 / 255, 185 / 255, 252 / 255))
        self.ax.spines["left"].set_color((141 / 255, 185 / 255, 252 / 255))
        self.ax.tick_params(colors=(141 / 255, 185 / 255, 252 / 255))

        self.ax.set_xlabel(f"{self.gene_name} Expression Level", fontsize=12, color=(141 / 255, 185 / 255, 252 / 255))
        self.ax.set_ylabel(f"Probability Distribution of {self.gene_name}", fontsize=12,
                           color=(141 / 255, 185 / 255, 252 / 255))

        # 设置右轴样式
        self.ax_right.set_facecolor((15 / 255, 29 / 255, 43 / 255))
        self.ax_right.spines["right"].set_color((255 / 255, 152 / 255, 0 / 255))
        self.ax_right.tick_params(colors=(255 / 255, 152 / 255, 0 / 255))

        self.ax_right.set_ylabel(f"Actual Probability Density of {self.gene_name}", fontsize=12,
                                 color=(255 / 255, 152 / 255, 0 / 255))

        # 明确告诉 matplotlib：右轴在右边
        self.ax_right.yaxis.set_label_position("right")
        self.ax_right.yaxis.tick_right()

        # 绘图部分...
        self.ax.plot(x, y, color=(141 / 255, 185 / 255, 252 / 255), lw=2)

        # 绘制橙色预测柯西分布曲线
        if new_x is not None and new_y is not None:
            if isinstance(new_x, torch.Tensor):
                new_x = new_x.detach().cpu().numpy()
            if isinstance(new_y, torch.Tensor):
                new_y = new_y.detach().cpu().numpy()

            alpha = 0.5
            gamma = max(0.3, alpha * (1 - new_y))
            cauchy_x = np.linspace(l_x, r_x, 300)
            cauchy_y = cauchy.pdf(cauchy_x, loc=new_x, scale=gamma)
            cauchy_y = (cauchy_y / max(cauchy_y)) * new_y
            self.ax_right.plot(cauchy_x, cauchy_y, color=(255 / 255, 152 / 255, 0 / 255), lw=2)
            self.ax_right.set_ylim(0, max(cauchy_y) * 1.2)

        # 添加预测文本标注
        self.ax.annotate(
            f"Level of the Prediction {self.gene_name} is {self.prediction}",
            xy=(0.5, -0.2),
            xycoords="axes fraction",
            fontsize=16,
            color=(141 / 255, 185 / 255, 252 / 255),
            ha="center",
            va="top"
        )

        self.draw()  # 刷新界面



from Script.Final_version_dl_4 import FeatureTransformer, PositionalEncoding, ResidualBlock, DeepRadiomicsClassifier

class RunPredict:
    def __init__(self,dia,data_path = "Script/Express_Visualize/test_example_data.csv"):
        pass
        data_path = "Script/Express_Visualize/test_example_data.csv"
        self.data_path = data_path
        self.dia = dia

    def prediction(self,best_model_path):
        RFs = ["log-sigma-5-0-mm-3D_firstorder_10Percentile",
                "log-sigma-5-0-mm-3D_glszm_SmallAreaHighGrayLevelEmphasis" ,
                "original_shape_Maximum2DDiameterColumn",
                "original_shape_Maximum2DDiameterRow",
                "original_shape_Maximum3DDiameter",
                "lbp-3D-m1_gldm_DependenceNonUniformity",
                "lbp-3D-m1_glszm_LargeAreaEmphasis",
                "lbp-3D-m2_gldm_DependenceNonUniformity",
                "lbp-3D-m2_glszm_LargeAreaEmphasis",
                "lbp-3D-m2_glszm_LargeAreaLowGrayLevelEmphasis",
                "square_glcm_Id",
                "wavelet-LLH_firstorder_Mean",
                "wavelet-HLH_glszm_LargeAreaEmphasis",
                "lbp-3D-k_glszm_HighGrayLevelZoneEmphasis"
               ]
        model = FeatureTransformer(input_dim=14, d_model=32, nhead=4,
                                   nhid=512, nlayers=3, dropout=0.1)
        try:
            model.load_state_dict(torch.load(best_model_path))
            model.eval()

            df = pd.read_csv(self.data_path)
            scaler = StandardScaler()
            input_data = torch.tensor(scaler.fit_transform(df[RFs].values), dtype=torch.float32)
            all_columns_exist = all(col in df.columns for col in RFs)
            if not all_columns_exist:
                print("Some columns are missing in the dataframe.")
                error_message = f"Some columns are missing in the dataframe.Please be sure all features have been extracted!"
                raise ValueError(error_message)
            with torch.no_grad():
                probabilities = model(input_data)
                predictions = (probabilities > 0.5).int()

            pred_result = predictions.tolist()
            probability = probabilities.tolist()

            print("Predicted Labels:", pred_result)
            print("Probabilities:", probability)

            return predictions, probabilities
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None, None

    def write_res(self,line_name,prediction=1,gene_name=None):
        if(gene_name == None):
            if (prediction == 1):
                line_name.setText("up - regulation")
            else:
                line_name.setText("down - regulation")
        else:
            ups = ['KYNU']
            if(gene_name in ups):
                line_name.setText("up - regulation")
            else:
                line_name.setText("down - regulation")

    def main(self):
        data = pd.read_csv("Script/Express_Visualize/TPM_CGGA.csv")
        LogNorm = ['TRAM2']#
        Gamma = ['CD40', 'KYNU', 'CCRL2', 'FCGR2A', 'JUNB', 'MBOAT1','FCGR2A']
        self.canvas_dict = {}
        #预测 画橙色和蓝色曲线
        for col_name in Gamma:
            best_model_path = f"Script/Express_Visualize/dl_model/best_model_{col_name}.pth"
            data_column = data[col_name]
            shape, loc, scale = stats.gamma.fit(data_column)
            x = np.linspace(min(data_column), max(data_column), 100)
            if (col_name == 'FCGR2A'):
                shape = 1.5737515419309507
                scale = 8022.627499650851
                q_low, q_high = np.percentile(data_column, [1, 98])
                x = np.linspace(q_low, q_high, 150)
            elif (col_name == 'CD40'):
                shape = 3.2443377626673273
                scale = 794.5733995413987
                # q_low, q_high = np.percentile(data_column, [5, 98])
                # x = np.linspace(q_low, q_high, 150)

            y = stats.gamma.pdf(x, shape, loc, scale)

            # 进行预测
            predictions, probabilities = self.prediction(best_model_path)
            if predictions is None or probabilities is None:
                continue
                # 动态获取对应的 gridLayout
            grid_layout_name = f"{col_name}_gridLayout_show_Expression_level"
            gridLayout = getattr(self.dia, grid_layout_name)
            line_name = f"{col_name}_lineEdit"
            line_name = getattr(self.dia, line_name)

            self.write_res(line_name,predictions,col_name)

            self.canvas_dict[col_name] = PlotCanvas((str)(col_name),predictions)
            gridLayout.addWidget(self.canvas_dict[col_name], 0, 0, 1, 1)
            # 数据转化
            max_level = max(data_column)
            min_level = min(data_column)
            x_new = probabilities * (max_level - min_level) + min_level
            # y_new = stats.gamma.pdf(x_new, shape, loc, scale)
            y_new = probabilities
            print(f"gene:{col_name}")
            print(f"x_new,y_new:{x_new},{y_new}")

            if(x_new <100):
                x_new = x_new*1000
            self.canvas_dict[col_name].plot_static_curve(x, y,x_new ,y_new, min_level, max_level)
            # break

        #画底坐标轴
        for col_name in LogNorm:
            # 拟合对数正态分布
            data_column = data[col_name]
            shape, loc, scale = lognorm.fit(data_column)
            if (col_name == 'TRAM2'):
                shape = 0.5713377136054707
                scale = 11226.136703445358
                q_low, q_high = np.percentile(data_column, [1, 98])
                x = np.linspace(q_low, q_high, 150)

            y = lognorm.pdf(x, shape, loc, scale)

            # 进行预测
            predictions, probabilities = self.prediction(best_model_path)
            grid_layout_name = f"{col_name}_gridLayout_show_Expression_level"
            gridLayout = getattr(self.dia, grid_layout_name)
            line_name = f"{col_name}_lineEdit"
            line_name = getattr(self.dia, line_name)

            self.write_res(line_name, predictions,col_name)
            self.canvas_dict[col_name] = PlotCanvas(str(col_name),predictions)
            gridLayout.addWidget(self.canvas_dict[col_name], 0, 0, 1, 1)

            if predictions is None or probabilities is None:
                continue

            # 数据转化
            max_level = max(data_column)
            min_level = min(data_column)
            x_new = probabilities * (max_level - min_level) + min_level
            y_new = probabilities
            print(f"gene:{col_name}")
            print(f"x_new,y_new:{x_new},{y_new}")

            self.canvas_dict[col_name].plot_static_curve(x, y, x_new, y_new, min_level, max_level)
            # break
