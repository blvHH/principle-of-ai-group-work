from sklearn.model_selection import cross_val_score
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
data_dir = r"C:\Users\吴\Desktop\code\python\principle\bmp"
image_size = (64, 64)
pca_components = 80
output_dir = R"C:\Users\吴\Desktop\code\python\principle"
os.makedirs(output_dir, exist_ok=True)


class ACOFeatureSelector:
    def __init__(self, X, y, num_ants=10, num_generations=20, evaporation_rate=0.3,
                 alpha=2, beta=1, top_k=300, save_dir="aco_logs", random_state=42,use_heuristic=True,resume=True):

        self.use_heuristic = use_heuristic
        self.X = X
        self.y = y
        self.num_ants = num_ants
        self.num_generations=num_generations
        self.evaporation_rate=evaporation_rate
        self.alpha=alpha
        self.beta=beta
        self.top_k=top_k
        self.random_state = random_state
        self.num_features = X.shape[1]
        self.pheromone = np.ones(self.num_features)
        self.save_dir = save_dir
        self.resume = resume
        self.best_features = None
        self.best_score = 0
        self.pheromone_path = os.path.join(self.save_dir, "pheromone.npy")
        self.best_features_path = os.path.join(self.save_dir, "aco_best_features.npy")
        os.makedirs(self.save_dir, exist_ok=True)
        np.random.seed(random_state)
        if resume and os.path.exists(self.pheromone_path):
            self.pheromone = np.load(self.pheromone_path)
            print("[INFO] Resumed pheromone from file.")
        else:
            self.pheromone = np.ones(self.num_features)
        self.use_heuristic = use_heuristic
        if self.use_heuristic:
            self.heuristic = np.var(self.X, axis=0)
            self.heuristic = np.nan_to_num(self.heuristic, nan=0.0)
        else:
            self.heuristic = np.ones(self.num_features)

    def evaluate(self, features):
        if len(features) == 0:
            return 0
        X_selected=self.X[:,features]
        '''
        pca=PCA(n_components=min(100,X_selected.shape[1]))
        X_pca=pca.fit_transform(X_selected)'''
        print("try_feature_length:",X_selected.shape[1])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(SVC(kernel='rbf', probability=True, class_weight='balanced'),X_selected, self.y, cv=cv).mean()
        print("finished with score:",score)
        return score

    def select_features(self, patience=10):
        if self.resume:
            self.best_features = np.load(r"C:\Users\吴\Desktop\code\python\principle\aco_logs\aco_best_features.npy", allow_pickle=True).tolist()
            print("加载成功，评估中...")
            self.best_score = self.evaluate(self.best_features)
            print(f"Resumed best features with score: {self.best_score:.4f}")
        else:
            return
            self.best_features = None
            self.best_score = 0
        best_gen=0
        generation_scores=[]

        for gen in range(self.num_generations):
            all_solutions = []
            all_scores = []
            for ant in range(self.num_ants):
                pheromone_term = self.pheromone ** self.alpha
                heuristic_term = self.heuristic ** self.beta
                prob = pheromone_term * heuristic_term
                prob /= prob.sum()
                chosen = np.random.choice(self.num_features, self.top_k, replace=False, p=prob)
                chosen = sorted(chosen)
                score = self.evaluate(chosen)
                all_solutions.append(chosen)
                all_scores.append(score)
                if score > self.best_score:
                    self.best_score = score
                    self.best_features = chosen
                    best_gen = gen  # 更新最后一次提升的 generation
            # 信息素更新
            self.pheromone *= (1 - self.evaporation_rate)
            for features, score in zip(all_solutions, all_scores):
                for f in features:
                    self.pheromone[f] += score
            # 保存每代结果
            np.save(os.path.join(self.save_dir, f"ant_generation_{gen}_selected.npy"), all_solutions)
            generation_scores.append({
                "generation": gen,
                "best_score": max(all_scores),
                "mean_score": np.mean(all_scores),
                "best_index": all_solutions[np.argmax(all_scores)]
            })
            print(
                f"Generation {gen}/{self.num_generations}, Best Score: {self.best_score:.4f}, selected:{len(all_solutions[np.argmax(all_scores)])}")
            # ---------- 早停判断 ----------
            if gen - best_gen >= patience:
                print(f"Early stopping at generation {gen } (no improvement in {patience} rounds)")
                break

        # 保存最终最佳索引与日志
        np.save(os.path.join(self.save_dir, "aco_best_features.npy"), self.best_features)
        pd.DataFrame(generation_scores).to_csv(os.path.join(self.save_dir, "generation_scores.csv"), index=False)

        return self.best_features, self.best_score


def load_and_preprocess_images(data_dir, image_size, save_path_X="X.npy", save_path_y="y.npy", reuse=True):
    if reuse and os.path.exists(save_path_X) and os.path.exists(save_path_y):
        print("已加载保存的数据文件")
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
    print(f" 图像数据和标签已保存到 {save_path_X} 和 {save_path_y}")
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
aco = ACOFeatureSelector(X_train_scaled, y_train, num_ants=10, num_generations=100, top_k=200)
print("进入蚁群搜索")
best_features, best_score = aco.select_features()

# 应用最优特征，训练并评估

X_selected = X_train_scaled[:, best_features]
X_test_selected = X_test_scaled[:, best_features] #只保留best_features里的位置
'''
pca = PCA(n_components=min(100, X_selected.shape[1]))
X_train_pca = pca.fit_transform(X_selected)
X_test_pca = pca.transform(X_test_selected)
'''
svm = SVC(kernel='rbf', probability=True,class_weight='balanced')
svm.fit(X_selected, y_train)
y_pred = svm.predict(X_test_selected)

print("Classification Report:")
print(classification_report(y_test, y_pred))
report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(output_dir, "ant_classification_report.csv"), index=True)


'''
import matplotlib.pyplot as plt
scores = pd.read_csv(os.path.join(self.save_dir, "generation_scores.csv"))
plt.plot(scores["generation"], scores["best_score"], label="Best Score")
plt.plot(scores["generation"], scores["mean_score"], label="Mean Score")
plt.legend()
plt.xlabel("Generation")
plt.ylabel("Score")
plt.title("ACO Progress")
plt.show()

'''