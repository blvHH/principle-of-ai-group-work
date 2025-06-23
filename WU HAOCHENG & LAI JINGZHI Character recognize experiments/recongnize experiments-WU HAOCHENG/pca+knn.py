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
pca_components = 100
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
joblib.dump(le, 'label_encoder.joblib')
for idx, label in enumerate(le.classes_):
    print(f"{idx} -> {label}")
# 3. 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
# 4. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'knn_scaler.joblib')
'''
# 5. PCA
print("开始PCA")
#pca = PCA(n_components=pca_components)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)'''
# 6. KNN 训练
print("开始训练 KNN")
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
knn.fit(X_train_scaled, y_train)
joblib.dump(knn, os.path.join(output_dir, "knnonly_model.joblib"))

# 7. 模型评估
y_pred = knn.predict(X_test_scaled)
report = classification_report(y_test, y_pred, target_names=le.classes_)
print(report)

# 8. 分类报告保存为 CSV
report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(output_dir, "KNNONLY_classification_report.csv"), index=True)