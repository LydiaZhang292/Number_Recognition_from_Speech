import torch
import torch.nn as nn
# 定义 6 层 1D-CNN 模型
class Emotion1DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(Emotion1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 , 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = self.activation(self.conv3(x))
        x = self.pool(x)
        x = self.activation(self.conv4(x))
        x = x.view(x.size(0), -1)  # 展平
        #x = torch.relu(self.fc1(x))
        x = self.activation(self.fc1(x))
        #x = self.dropout(x
        x = self.fc2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()

        # 如果输入和输出通道数不同，使用1x1卷积进行通道对齐
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        return self.relu(x)

class ResNet1D(nn.Module):
    def __init__(self, num_classes):
        super(ResNet1D, self).__init__()
        self.layer1 = nn.Sequential(
            #nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False), 这里参考以上channel=1
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.resblock1 = ResidualBlock(64, 64)
        self.resblock2 = ResidualBlock(64, 128)
        self.resblock3 = ResidualBlock(128, 128)
        self.resblock4 = ResidualBlock(128, 256)
        self.resblock5 = ResidualBlock(256, 256)
        self.resblock6 = ResidualBlock(256, 512)
        self.resblock7 = ResidualBlock(512, 512)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化，输出大小为 (batch_size, channels, 1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平为 (batch_size, 512)
        x = self.fc(x)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, seq_len, d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # 1. 输入特征线性变换到 d_model 维度
        self.feature_embedding = nn.Linear(input_dim, d_model)

        # 2. 位置编码
        self.position_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # 3. Transformer 编码器
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,  # 使输入为 (batch_size, seq_len, d_model)
            ),
            num_layers=num_encoder_layers
        )

        # 4. 全局池化层（取序列维度的均值）
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 5. 分类头
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()

        # 特征线性变换到 d_model
        x = self.feature_embedding(x)  # (batch_size, seq_len, d_model)

        # 添加位置编码
        x = x + self.position_encoding[:, :seq_len, :]

        # Transformer 编码
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

        # 全局池化 (平均池化 over seq_len)
        x = x.transpose(1, 2)  # 转换为 (batch_size, d_model, seq_len) 以便池化
        x = self.global_avg_pool(x)  # (batch_size, d_model, 1)
        x = x.squeeze(-1)  # (batch_size, d_model)

        # 分类头
        x = self.fc(x)  # (batch_size, num_classes)
        return x