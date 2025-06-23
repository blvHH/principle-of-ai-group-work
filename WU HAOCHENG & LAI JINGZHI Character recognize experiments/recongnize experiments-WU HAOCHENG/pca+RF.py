from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC,LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

data_dir = r"C:\Users\吴\Desktop\code\python\principle\bmp"
image_size = (64, 64)
output_dir = R"C:\Users\吴\Desktop\code\python\principle"
os.makedirs(output_dir, exist_ok=True)
# ---------------------------------

def load_and_preprocess_images(data_dir, image_size, save_path_X="X.npy", save_path_y="y.npy", reuse=True):
    # 如果已存在保存的文件且允许复用，直接加载
    if reuse and os.path.exists(save_path_X) and os.path.exists(save_path_y):
        print("✅ 已加载保存的数据文件")
        return np.load(save_path_X), np.load(save_path_y)

    # 否则进行读取和处理
    X, y = [], []
    flag=1
    for label in sorted(os.listdir(data_dir)):
        label_folder = os.path.join(data_dir, label)
        if not os.path.isdir(label_folder):
            continue
        for img_name in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, image_size)
            if flag:
                print(img_resized.flatten())
                print(label)
                flag=0
            X.append(img_resized.flatten())
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    # 保存结果
    np.save(save_path_X, X)
    np.save(save_path_y, y)
    print(f"✅ 图像数据和标签已保存到 {save_path_X} 和 {save_path_y}")
    return X, y

# 1. 加载数据
print("开始读取数据")
X, y = load_and_preprocess_images(data_dir, image_size)

# 2. 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
# 4. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
'''
# 5. PCA
print("开始PCA")
#pca = PCA(n_components=pca_components)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
'''
from sklearn.ensemble import RandomForestClassifier

# 6. 随机森林训练
print("开始训练 Random Forest")
rf = RandomForestClassifier(
    n_estimators=300,           # 建议提高到 200~500，更多树提升鲁棒性（代价是训练时间）
    max_depth=30,               # 控制模型复杂度，避免过拟合（如你的特征是PCA后的，可选15~40）
    min_samples_split=4,        # 限制每棵树中最小分裂样本数（可防止太深）
    min_samples_leaf=2,         # 每片叶子最少样本，缓解不平衡问题
    max_features='sqrt',        # 每次分裂考虑特征数量（默认为"sqrt"，也可尝试"log2"）
    class_weight='balanced',    # 对不平衡类别加权（已正确设置）
    bootstrap=True,             # 启用自助采样法，提高泛化能力
    random_state=42,            # 保持可复现
    n_jobs=-1,                  # 启用所有线程
    verbose=1                   # 输出训练过程，方便监控
)
rf.fit(X_train_scaled, y_train)
joblib.dump(rf, os.path.join(output_dir, "rfonly_model.joblib"))

# 7. 模型评估
y_pred = rf.predict(X_test_scaled )
report = classification_report(y_test, y_pred, target_names=le.classes_)
print(report)

# 8. 分类报告保存为 CSV
report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(output_dir, "RFonly_classification_report.csv"), index=True)
