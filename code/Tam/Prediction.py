
from Tam.Final_version_dl_4 import FeatureTransformer, PositionalEncoding, ResidualBlock, DeepRadiomicsClassifier
import torch
import pandas as pd
import numpy as np
from scipy.stats import cauchy
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
        self.ax.spines['bottom'].set_color((141/255, 185/255, 252/255))
        self.ax.spines['left'].set_color((141/255, 185/255, 252/255))
        self.ax.tick_params(colors=(141/255, 185/255, 252/255))
        self.ax.set_xlabel(f"{self.title} GSVA Score", fontsize=12, color=(141/255, 185/255, 252/255))
        self.ax.set_ylabel(f"Density of {self.title}", fontsize=12, color=(141/255, 185/255, 252/255))

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

            alpha = 0.5
            gamma = max(0.3, alpha * (1 - new_y))
            cauchy_x = np.linspace(l_x, r_x, 300)
            cauchy_y = cauchy.pdf(cauchy_x, loc=new_x, scale=gamma)
            cauchy_y = (cauchy_y / max(cauchy_y)) * new_y
            self.ax_right.plot(cauchy_x, cauchy_y, color=(255/255, 152/255, 0/255), lw=2, label="Cauchy Peak")
            self.ax_right.set_ylim(0, max(cauchy_y)*1.2)

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


class TamPredict:
    def __init__(self, dia, data_path="Tam/new_cgga_gsva.csv"):
        self.dia = dia
        self.data_path = data_path

    def prediction(self, best_model_path):
        RFs = [
            "log-sigma-5-0-mm-3D_firstorder_10Percentile",
            "log-sigma-5-0-mm-3D_glszm_SmallAreaHighGrayLevelEmphasis",
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
        model = FeatureTransformer(input_dim=14, d_model=32, nhead=4, nhid=512, nlayers=3, dropout=0.1)
        try:
            model.load_state_dict(torch.load(best_model_path))
            model.eval()
            df = pd.read_csv(self.data_path)
            scaler = StandardScaler()
            input_data = torch.tensor(scaler.fit_transform(df[RFs].values), dtype=torch.float32)
            if not all(col in df.columns for col in RFs):
                raise ValueError("Missing required feature columns in input file.")
            with torch.no_grad():
                probs = model(input_data)
                preds = (probs > 0.5).int()
            return preds, probs
        except Exception as e:
            print("Error during prediction:", e)
            return None, None

    def write_res(self, line_widget, prediction):
        if prediction == 1:
            line_widget.setText("up")
        else:
            line_widget.setText("down")

    def main(self):
        data = pd.read_csv("Tam/new_cgga_gsva.csv")
        cluster_name = "cluster1"
        gsva_scores = data[cluster_name]
        kde = gaussian_kde(gsva_scores)
        x = np.linspace(min(gsva_scores), max(gsva_scores), 200)
        y = kde(x)

        # 做预测
        best_model_path = "Script/Express_Visualize/dl_model/best_model_cluster1.pth"
        preds, probs = self.prediction(best_model_path)
        if preds is None or probs is None:
            return

        gridLayout = getattr(self.dia, "tam_gridLayout")
        lineEdit = getattr(self.dia, "tam_lineEdit")

        self.write_res(lineEdit, int(preds[0]))

        self.canvas = PlotCanvas("cluster1", int(preds[0]))
        gridLayout.addWidget(self.canvas, 0, 0, 1, 1)

        min_level = min(gsva_scores)
        max_level = max(gsva_scores)
        x_new = probs[0] * (max_level - min_level) + min_level
        y_new = probs[0]

        self.canvas.plot_static_curve(x, y, x_new, y_new, min_level, max_level)
