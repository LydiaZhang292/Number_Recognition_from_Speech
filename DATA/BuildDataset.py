import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def process_deap_data(csv_file, output_file=None):
    """
    从DEAP数据CSV文件中加载数据，添加情感标签，调整数据形状，划分训练集和测试集。

    Args:
        csv_file (str): 输入的CSV文件路径。
        output_file (str, optional): 如果提供，保存处理后的数据到该文件。

    Returns:
        tuple: (X_train, X_test, y_train, y_test)，分别是训练集和测试集的特征和标签。
    """
    # 加载CSV文件
    data = pd.read_csv(csv_file)

    # 检查数据列数
    if data.shape[1] < 16:
        raise ValueError("CSV文件的列数不足，必须包含前12列特征和后4列指标（Valence, Arousal, Dominance, Liking）。")

    # 提取特征和目标
    features = data.iloc[:, :12].values  # 前12列为特征
    valence = data.iloc[:, 14].values  # Valence 列
    arousal = data.iloc[:, 15].values  # Arousal 列
    #dominace = data.iloc[:, 16].values  # dominance 列

    # 添加情感标签：根据 Valence 和 Arousal 的高低划分四类情感
    labels = []
    for v, a in zip(valence, arousal):
        if v > 5 and a > 5:
            labels.append(0)  # 高兴
        elif v > 5 and a <= 5:
            labels.append(1)  # 放松
        elif v <= 5 and a > 5:
            labels.append(2)  # 愤怒
        else:
            labels.append(3)  # 悲伤
    labels = np.array(labels)

    # 标准化特征
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 调整形状为 (样本数, 时间步数, 通道数)
    features = np.expand_dims(features, axis=-1)  # 添加通道维度，形状变为 (样本数, 12, 1)

    # 划分训练集和测试集（4:1 比例）
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 如果指定了输出文件，保存数据
    if output_file:
        np.savez(output_file, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        print(f"处理后的数据已保存到 {output_file}")

    return X_train, X_test, y_train, y_test


csv_path = "E:\\Homework\\2024FW\\DSP\\EXP3\\RAW_DATA\\features.csv"  # 替换为你的CSV文件路径
output_path = "E:\\Homework\\2024FW\\DSP\\EXP3\\DATA\\features.csv" # 输出保存的文件名

X_train, X_test, y_train, y_test = process_deap_data(csv_path, output_file=output_path)

# 打印数据形状以验证
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
