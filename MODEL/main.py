import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import Model

batch_size = 32

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#加载数据
def load_dataset(npz_file):
    data = np.load(npz_file)
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']

    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)  # (样本数, 1, 12)
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)    # (样本数, 1, 12)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    print("Training complete!")

# 测试模型
def test_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")

# 主程序
if __name__ == "__main__":
    # 数据加载
    npz_file = "E:\\Homework\\2024FW\\DSP\\EXP3\\DATA\\features.csv.npz"
    X_train, X_test, y_train, y_test = load_dataset(npz_file)

    # 数据集与 DataLoader

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model=Model.TransformerClassifier(input_dim=12,num_classes=4,seq_len=128)
    criterion = nn.CrossEntropyLoss()
    #class_weights = torch.tensor([1.0, 0.5, 0.8, 1.2]).to(device)
    #criterion = nn.CrossEntropyLoss(weight=class_weights)

    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 训练模型
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=20)

    # 测试模型
    test_model(model, test_loader)