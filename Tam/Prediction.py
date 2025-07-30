
from Tam.Final_version_dl_4 import GoogleNetV2, FeatureTransformer, PositionalEncoding, ResidualBlock, DeepRadiomicsClassifier
import torch
import pandas as pd
import numpy as np
from scipy.stats import cauchy
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas



class PlotCanvas(FigureCanvas):
    def __init__(self, title, prediction=1, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.title = str(title)
        self.prediction = int(prediction)

        # 主轴（左轴）设置
        self.fig.patch.set_facecolor((15/255, 29/255, 43/255))
        self.ax.set_facecolor((15/255, 29/255, 43/255))
        # 显式设置左轴和下轴线条颜色和宽度
        self.ax.spines['left'].set_visible(True)
        self.ax.spines['left'].set_color((141 / 255, 185 / 255, 252 / 255))
        self.ax.spines['left'].set_linewidth(1.5)

        self.ax.spines['bottom'].set_visible(True)
        self.ax.spines['bottom'].set_color((141 / 255, 185 / 255, 252 / 255))
        self.ax.spines['bottom'].set_linewidth(1.5)
        self.ax.tick_params(colors=(141/255, 185/255, 252/255))
        self.ax.set_xlabel(f"{self.title} GSVA Score", fontsize=12, color=(141/255, 185/255, 252/255))
        self.ax.set_ylabel(f"Density of {self.title}", fontsize=12, color=(141/255, 185/255, 252/255))
        self.ax.set_xlim(-1, 1)

        # 右轴设置
        self.ax_right = self.ax.twinx()
        self.ax_right.spines['right'].set_color((255/255, 152/255, 0/255))
        self.ax_right.tick_params(colors=(255/255, 152/255, 0/255))
        self.ax_right.set_ylabel(f"Predicted Confidence", fontsize=12, color=(255/255, 152/255, 0/255))

    def plot_static_curve(self, x, y, new_x=None, new_y=None, l_x=None, r_x=None):
        self.ax.plot(x, y, color=(141/255, 185/255, 252/255), lw=2, label="KDE Distribution")

        if new_x is not None and new_y is not None:
            if isinstance(new_x, torch.Tensor):
                new_x = new_x.detach().cpu().numpy()
            if isinstance(new_y, torch.Tensor):
                new_y = new_y.detach().cpu().numpy()

            # 自定义尖峰高度：越接近0或1越高，越接近0.5越低
            confidence = abs(new_y - 0.5)  # 越接近 0.5，值越小
            height = 10 ** (2 + 2 * confidence)  # 可调：最小约 10^2，最大约 10^4

            # 设置横轴范围
            cauchy_x = np.linspace(l_x, r_x, 300)
            cauchy_y = np.zeros_like(cauchy_x)

            # 找到 new_x 最近的点位置，画出尖峰
            peak_index = np.argmin(np.abs(cauchy_x - new_x))
            cauchy_y[peak_index] = height

            # 绘图
            self.ax_right.plot(cauchy_x, cauchy_y, color=(255 / 255, 152 / 255, 0 / 255), lw=2,
                               label="Confidence Spike")
            self.ax_right.set_ylim(0, height * 1.2)  # 上限稍高以显示完整


        self.ax.annotate(
            f"Prediction: {self.title} → {self.prediction}",
            xy=(0.5, -0.2),
            xycoords="axes fraction",
            fontsize=16,
            color=(141/255, 185/255, 252/255),
            ha="center",
            va="top"
        )
        self.draw()
        # 保存图像到 result 目录（确保目录已存在）
        print("已绘制图")
        self.fig.savefig(f"result/angiotam_prediction_plot.png", dpi=300, bbox_inches='tight') #tight用于去除白边


class TamPredict:
    def __init__(self, dia, data_path="Tam/data/test_sample2.csv"):
        '''self.dia = dia
        if isinstance(data_path, list):
            data_path = data_path[0]
        self.data_path = data_path'''
        '''pass
        data_path = "result/a.csv"'''
        data_path = data_path[0]
        self.data_path = data_path
        self.dia = dia

    def prediction(self, best_model_path):
        RFs = ["log-sigma-3-mm-3D_firstorder_Maximum",
               "log-sigma-3-mm-3D_glcm_ClusterShade",
               "wavelet-HLL_firstorder_Skewness",
               "wavelet-HLL_glcm_ClusterShade",
               "wavelet-HLL_glcm_Correlation",
               "exponential_firstorder_Kurtosis",
               "wavelet-HHL_glcm_ClusterShade",
               "wavelet-HHL_glszm_LargeAreaHighGrayLevelEmphasis",
               "gradient_glcm_Correlation",
               "original_shape_Sphericity",
               "wavelet-HLL_firstorder_Mean",
               "log-sigma-3-mm-3D_firstorder_Skewness",
               "wavelet-HLL_firstorder_RootMeanSquared",
               "wavelet-HLH_glcm_ClusterShade",
               "wavelet-LHH_glcm_Imc1",
               "wavelet-HHH_firstorder_Skewness",
               "wavelet-HHH_firstorder_Maximum",
               "wavelet-LHH_firstorder_Skewness",
               "wavelet-LLH_gldm_LargeDependenceLowGrayLevelEmphasis",
               "wavelet-LHH_glcm_Imc2"
               ]
        model = GoogleNetV2(input_dim=20)
        print("已设置模型为 GoogleNetV2")

        try:
            model.load_state_dict(torch.load(best_model_path, map_location=torch.device("cpu")))
            print("已读取最佳模型")
            model.eval()

            df = pd.read_csv(self.data_path, header=None)  # 不自动用第一行做列名
            df = df[1:].set_index(0).T  # 去掉表头后，将第0列设置为index，然后转置
            print("转置成功")

            if not all(col in df.columns for col in RFs):
                raise ValueError("Missing required feature columns in input file.")

            scaler = StandardScaler()
            input_data = torch.tensor(scaler.fit_transform(df[RFs].values), dtype=torch.float32)
            #scaler = joblib.load("trained_scaler2.pkl")
            #input_data = torch.tensor(scaler.transform(df[RFs].values), dtype=torch.float32)

            print("提取特征完毕")

            with torch.no_grad():
                logits = model(input_data)
                probs = torch.sigmoid(logits).squeeze()
                preds = (probs > 0.5).int()

            print("预测完毕")
            print("Predicted Labels:", preds.tolist())
            print("Probabilities:", probs.tolist())

            return preds, probs

        except Exception as e:
            print("Error during prediction:", str(e))
            return None, None

    def write_res(self, line_widget, prediction):
        if prediction == 1:
            line_widget.setText("up - regulation")
        else:
            line_widget.setText("down - regulation")

    def write_res1(self, text_widget, prediction):
        if prediction == 1:
            text_widget.setHtml("This patient was identified as:<br>"
                                "<b><span style='color:red;'>High</span></b> enrichment degree of Angio-tams in the tumor area;<br>"
                                "<b><span style='color:red;'>High</span></b> risk of survival."
                                )
        else:
            text_widget.setHtml(
                "This patient was identified as:<br>"
                "<b><span style='color:green;'>Low</span></b> enrichment degree of Angio-tams in the tumor area;<br>"
                "<b><span style='color:green;'>Low</span></b> risk of survival."
            )

    def main(self):
        data = pd.read_csv("Tam/data/new_cgga_gsva.csv")
        cluster_name = "cluster1"
        gsva_scores = data[cluster_name]
        kde = gaussian_kde(gsva_scores)
        x = np.linspace(min(gsva_scores), max(gsva_scores), 200)
        y = kde(x)

        # 做预测
        best_model_path = "Tam/data/best_model_507.pth"
        preds, probs = self.prediction(best_model_path)
        if preds is None or probs is None:
            print("预测失败")
            return
        print("完成预测")
        print(f"预测结果为：preds:{preds},probs:{probs}")

        gridLayout = getattr(self.dia, "tam_gridLayout")
        print("完成gridLayout这行")
        lineEdit = getattr(self.dia, "tam_lineEdit")
        print("完成lineEdit这行")
        self.write_res(lineEdit, preds)
        print("完成self.write_res这行")
        textEdit = getattr(self.dia, "textEdit")
        self.write_res1(textEdit, preds)
        print("完成textEdit文本编辑")

        self.canvas = PlotCanvas("Angio-TAMs", preds)
        gridLayout.addWidget(self.canvas, 0, 0, 1, 1)
        print("完成self.canvas这行，绘图初始化完成")

        min_level = min(gsva_scores)
        max_level = max(gsva_scores)
        x_new = probs * (max_level - min_level) + min_level
        y_new = probs
        print(f"x_new,y_new:{x_new},{y_new}")


        self.canvas.plot_static_curve(x, y, x_new, y_new, min_level, max_level)
        print("完成self.canvas.plot_static_curve这行，绘图完成")
