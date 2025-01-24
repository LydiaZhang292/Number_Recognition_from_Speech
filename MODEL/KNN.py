import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


# 加载数据
def load_data(csv_path):
    """
    加载 CSV 数据，假设前 12 列是特征，后 4 列中 Valence 和 Arousal 用于情感分类
    """
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    features = data[:, :12]  # 前 12 列是特征
    labels = data[:, 14:16]  # 后 4 列中的 Valence 和 Arousal
    return features, labels


# 将 Valence 和 Arousal 转换为四分类标签
def generate_labels(valence_arousal):
    """
    根据 Valence 和 Arousal 值，将样本划分为四分类：
      - 0: 低 Valence，低 Arousal
      - 1: 低 Valence，高 Arousal
      - 2: 高 Valence，低 Arousal
      - 3: 高 Valence，高 Arousal
    """
    valence, arousal = valence_arousal[:, 0], valence_arousal[:, 1]
    labels = np.zeros(len(valence), dtype=int)
    labels[(valence >= 5) & (arousal >= 5)] = 3
    labels[(valence >= 5) & (arousal < 5)] = 2
    labels[(valence < 5) & (arousal >= 5)] = 1
    # 低 Valence 且低 Arousal 的标签默认是 0
    return labels


# 主函数
def main(csv_path):
    # 1. 加载数据
    features, valence_arousal = load_data(csv_path)

    # 2. 生成四分类标签
    labels = generate_labels(valence_arousal)

    # 3. 数据标准化
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 4. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 5. 训练 KNN 模型
    knn = KNeighborsClassifier(n_neighbors=5)  # 设置 k=5
    knn.fit(X_train, y_train)

    # 6. 模型测试
    y_pred = knn.predict(X_test)

    # 7. 输出结果
    print("分类报告:")
    print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1", "Class 2", "Class 3"]))

    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))


# 执行程序
if __name__ == "__main__":
    csv_path = "E:\\Homework\\2024FW\\DSP\\EXP3\\RAW_DATA\\features.csv"
    main(csv_path)
