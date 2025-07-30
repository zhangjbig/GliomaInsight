import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=15):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)

class FeatureTransformer(nn.Module):
    def __init__(self, input_dim=14, d_model=32, nhead=4, nhid=512, nlayers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.feature_embed = nn.Linear(1, d_model)  # 每个特征映射到d_model维
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        
        # 深度残差模块
        self.res_blocks = nn.Sequential(
            ResidualBlock(d_model, 128),  
            ResidualBlock(128, 256),     
            ResidualBlock(256, 128)      
        )
        
        # 注意力池化 
        self.attention_pool = nn.Sequential(
            nn.Linear(128, 64),          
            nn.Tanh(),
            nn.Linear(64, 1),            
            nn.Softmax(dim=1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            #nn.LayerNorm(d_model),
            nn.Linear(128, 64),          # 输入128，输出64
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),            # 输入64，输出1
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim=14)
        x = x.unsqueeze(-1)  # (batch_size, 14, 1)
        x = self.feature_embed(x) * math.sqrt(self.d_model)  # (batch_size, 14, d_model)
        x = self.pos_encoder(x)  # (batch_size, 14, d_model)
        
        transformer_out = self.transformer_encoder(x)  # (batch_size, 14, d_model)
        
        res_out = self.res_blocks(transformer_out.mean(dim=1))  # (batch_size, 128)
        
        # 注意力池化
        attn_weights = self.attention_pool(res_out)  # (batch_size, 1)
        weighted_features = res_out * attn_weights  # (batch_size, 128)
                
        return self.classifier(weighted_features).squeeze()  # (batch_size,)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, x):
        return nn.functional.gelu(self.block(x) + self.shortcut(x))

# 使用示例
class DeepRadiomicsClassifier(FeatureTransformer):
    def __init__(self):
        super().__init__(input_dim=14, d_model=32, nhead=4, 
                        nhid=512, nlayers=3, dropout=0.1)

if __name__ == "__main__":
    # 读取数据
    data_train = pd.read_csv("CGGA_tag.csv")
    X = data_train.iloc[:, 4:18].values
    X_train = X

    data_test = pd.read_csv("nnTCGA_tag.csv")
    X_test = data_test.iloc[:, 4:18].values

    col_to_pred = ['CCRL2','CD40','JUNB','KYNU','MBOAT1','TRAM2']
    for col_name in col_to_pred:
        print(f"正在处理列: {col_name}")
        os_raw = data_train[col_name]#.iloc[:, 18].values
        y_train = (os_raw == 2).astype(int)
        try:
            os_raw = data_test[col_name]#.iloc[:, 18].values
        except KeyError:
            continue
        y_test = (os_raw == 2).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y_train, test_size=0.3, random_state=17)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long).to(device)
        # y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

        # 创建DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DeepRadiomicsClassifier().to(device)
        # optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        optimizer = optim.RMSprop(model.parameters(), lr=0.0003, alpha=0.9, weight_decay=1e-4)
        max_grad_norm = 12 # 梯度裁剪，可调整梯度限制大小
        criterion = nn.BCELoss()
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)#StepLR(optimizer, step_size=10, gamma=0.7)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

        # 训练模型
        epochs = 10000
        train_losses = []
        # 训练模型
        best_val_acc = 0 #float('inf')  # 记录最优 validation loss
        best_epoch = -1  # 记录最佳 epoch
        best_model_path = f"best_model_{col_name}.pth"  # 存储最优模型路径

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            # train_losses.append(avg_train_loss)
            if avg_train_loss < 0.05:
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        outputs = model(X_batch)
                        predicted = (outputs > 0.5).long()
                        correct += (predicted == y_batch).sum().item()
                        total += y_batch.size(0)

                if correct / total > best_val_acc:
                    best_val_acc = correct / total
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), best_model_path)
        print(f"    Training complete. Best model at epoch {best_epoch} with val ACC {100*best_val_acc:.4f}%")

        # def moving_average(data, window_size):
        #     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        # window_size = 20
        # smooth_train_losses = moving_average(train_losses, window_size)
        # epochs_smooth = np.arange(window_size, epochs + 1)
        # plt.plot(epochs_smooth, smooth_train_losses, label='Train Loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.title(f'Training Loss Over Epochs of {col_name}')
        # plt.legend()
        # plt.show()
        
        break