import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# 建立交叉熵模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 生成单个样本
def build_sample():
    x = np.random.randn(5)
    max_x_id = np.argmax(x)  # 求出x向量中最大值的索引
    return x, max_x_id

# 生成数据集
def build_dataset(total_sample_num):
    X, Y = [], []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 评估模型
def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        accuracy = (torch.argmax(y_pred, dim=1) == y).float().mean().item()
    return accuracy

# 主训练函数
def main():
    # 超参数设置
    epoch_num = 30
    batch_size = 32
    train_sample = 5000
    input_size = 5
    learning_rate = 0.001
    
    # 创建模型和优化器
    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 生成数据并划分训练集和验证集
    X, Y = build_dataset(train_sample)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    # 记录训练历史
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    
    # 训练过程
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    
    for epoch in range(epoch_num):
        model.train()
        train_loss = 0.0
        
        # 批次训练
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i+batch_size]
            y_batch = Y_train[i:i+batch_size]
            
            optim.zero_grad()
            loss = model(x_batch, y_batch)
            loss.backward()
            optim.step()
            
            train_loss += loss.item() * len(x_batch)
        
        # 计算平均损失
        train_loss /= len(X_train)
        train_loss_history.append(train_loss)
        
        # 验证模型
        val_loss = model(X_val, Y_val).item()
        val_loss_history.append(val_loss)
        
        val_acc = evaluate(model, X_val, Y_val)
        val_acc_history.append(val_acc)
        
        # 学习率调整
        scheduler.step(val_loss)

        
        print(f"进行轮次 {epoch+1}/{epoch_num} - "
              f"平均损失: {val_loss:.4f}, "
              f"准确率: {val_acc:.4f}")


if __name__ == "__main__":
    main()