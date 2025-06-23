import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import pickle
from typing import Dict, List, Tuple, Callable
import copy
from collections import Counter
from itertools import product
from pathlib import Path
import cv2
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
knn_model = joblib.load(r"C:\Users\吴\Desktop\code\python\principle\knnonly_model.joblib")
scaler = joblib.load('knn_scaler.joblib')
le = joblib.load('label_encoder.joblib')
data_dir = r"C:\Users\吴\Desktop\code\python\principle\bmp"
image_size = (64, 64)
count=0

def load_and_preprocess_images(data_dir, image_size, save_path_X="X_clu.npy", save_path_y="y_clu.npy", reuse=True):
    if reuse and os.path.exists(save_path_X) and os.path.exists(save_path_y):
        return np.load(save_path_X), np.load(save_path_y)
    X, y = [], []
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
            X.append(img_resized)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    np.save(save_path_X, X)
    np.save(save_path_y, y)
    print(f"✅ 图像数据和标签已保存到 {save_path_X} 和 {save_path_y}")
    return X, y


def extract_features(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    contrast = img.std()
    laplacian = cv2.Laplacian(img, cv2.CV_64F).var()
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    orientation = np.arctan2(sobely.mean(), sobelx.mean())
    return np.array([contrast, laplacian, orientation])


def apply_preprocessing(img: np.ndarray, params: Dict) -> np.ndarray:
    proc_img = img.copy()

    if 'rot' in params:
        h, w = proc_img.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), params['rot'], 1)
        proc_img = cv2.warpAffine(proc_img, M, (w, h))

    if 'scale' in params:
        proc_img = cv2.resize(proc_img, None, fx=params['scale'], fy=params['scale'])
        proc_img = cv2.resize(proc_img, img.shape[::-1])
    if 'alpha' in params:
        proc_img = cv2.convertScaleAbs(proc_img, alpha=params['alpha'], beta=0)

    if 'sharp' in params:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * params['sharp']
        proc_img = cv2.filter2D(proc_img, -1, kernel)

    return proc_img

'''
def score_image(img: np.ndarray, template: np.ndarray) -> float:
    """使用模板匹配评估图像质量，自动处理类型和尺寸"""
    # 转灰度
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if template.ndim == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # 类型转为uint8
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)
    if template.dtype != np.uint8:
        template = cv2.convertScaleAbs(template)

    # 将模板resize为和图像一致大小
    if template.shape != img.shape:
        template = cv2.resize(template, (img.shape[1], img.shape[0]))

    # 执行模板匹配
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    return float(np.max(result))
'''
'''
def bfs_search_preprocess(
        img: np.ndarray,
        template: np.ndarray,
        init_params: Dict = None,
        steps: List[Dict] = None
) -> Tuple[Dict, np.ndarray, float]:
    """分步优化的BFS搜索最佳预处理参数，增加细粒度"""
    if init_params is None:
        init_params = {'alpha': 1.0, 'rot': 0.0, 'scale': 1.0, 'sharp': 1.0}
    if steps is None:
        steps = [
            {'param': 'rot', 'values': np.arange(-15.0, 15.1, 3.0)},  # 旋转：-15到15度，步长3度
            {'param': 'scale', 'values': np.arange(0.7, 1.31, 0.1)},  # 缩放：0.7到1.3，步长0.1
            {'param': 'alpha', 'values': np.arange(0.7, 1.31, 0.1)},  # 对比度：0.7到1.3，步长0.1
            {'param': 'sharp', 'values': np.arange(0.7, 1.31, 0.1)}  # 锐化：0.7到1.3，步长0.1
        ]

    best_score = -1
    best_proc = None
    best_params = init_params.copy()

    current_params = init_params.copy()

    for step in steps:
        param_queue = []
        param_key = step['param']
        for value in step['values']:
            temp_params = current_params.copy()
            temp_params[param_key] = float(value)  # 确保值为浮点数
            param_queue.append(temp_params)

        # 评估当前步骤
        for params in param_queue:
            proc = apply_preprocessing(img, params)
            score = score_image(proc, template)

            if score > best_score:
                best_score = score
                best_proc = proc.copy()
                best_params = params.copy()

        # 更新当前参数为本步骤最佳参数
        current_params = best_params.copy()

    return best_params, best_proc, best_score
'''

def score_image(img: np.ndarray, true_label: str) -> float:
    global count
    count += 1
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f",
               "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

    proc_img_resized = cv2.resize(img, (64, 64))
    feat = proc_img_resized.flatten().reshape(1, -1)
    feat = scaler.transform(feat)
    pred_index = knn_model.predict(feat)[0]
    predicted_category = le.inverse_transform([pred_index])[0]

    if predicted_category == true_label:
        distances, _ = knn_model.kneighbors(feat, n_neighbors=1)
        score = np.exp(-distances[0][0])
        return score
    else:
        return 0.0


def full_bfs_search(img, label):
    '''
    steps = [
        {'param': 'rot', 'values': np.arange(-15.0, 15.1, 3.0)},  # 旋转：-15到15度，步长3度
        {'param': 'scale', 'values': np.arange(0.8, 1.21, 0.1)},  # 缩放：0.7到1.3，步长0.1
        {'param': 'alpha', 'values': np.arange(0.8, 1.21, 0.1)},  # 对比度：0.7到1.3，步长0.1
        {'param': 'sharp', 'values': np.arange(0.8, 1.21, 0.1)}  # 锐化：0.7到1.3，步长0.1
    ]'''
    steps = [
        {'param': 'rot', 'values': [-10.0, 0.0, 10.0]},
        {'param': 'scale', 'values': [0.8, 1.0, 1.2]},
        {'param': 'alpha', 'values': [0.8, 1.0, 1.2]},
        {'param': 'sharp', 'values': [0.8, 1.0, 1.2]}
    ]

    value_lists = [step['values'] for step in steps]
    param_keys = [step['param'] for step in steps]
    best_score = -1
    best_proc = None
    best_params = None

    for values in product(*value_lists):
        params = {k: float(v) for k, v in zip(param_keys, values)}
        proc = apply_preprocessing(img, params)
        score = score_image(proc, label)
        if score > best_score:
            best_score = score
            best_proc = proc.copy()
            best_params = params.copy()
    #print()
    return best_params, best_proc, best_score


def vote_best_params(param_list: List[Dict]) -> Dict:
    if not param_list:
        return {'alpha': 1.0, 'rot': 0.0, 'scale': 1.0, 'sharp': 1.0}

    discretized_params = []
    for params in param_list:
        disc_params = (
            round(params['rot'] / 3.0) * 3.0,
            round(params['scale'] / 0.1) * 0.1,
            round(params['alpha'] / 0.1) * 0.1,
            round(params['sharp'] / 0.1) * 0.1
        )
        discretized_params.append(disc_params)
    counter = Counter(discretized_params)
    most_common = counter.most_common(1)[0][0]
    return {
        'rot': most_common[0],
        'scale': most_common[1],
        'alpha': most_common[2],
        'sharp': most_common[3]
    }
def cluster_and_optimize(
        data_dir: str,
        n_clusters: int = 5,
        samples_per_cluster: int = 20
) -> Tuple[Dict, KMeans, List[float]]:

    images, labels = load_and_preprocess_images(data_dir,image_size)
    features = np.array([extract_features(img) for img in images])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    cluster_params = {}
    all_scores = []

    for cluster in range(n_clusters):
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
        if not cluster_indices:
            continue

        selected_indices = np.random.choice(
            cluster_indices, min(samples_per_cluster, len(cluster_indices)), replace=False
        )
        selected_images = [images[i] for i in selected_indices]
        selected_labels = [labels[i] for i in selected_indices]

        param_list = []
        cluster_scores = []

        for img, label in zip(selected_images, selected_labels):
            best_params, _, score = full_bfs_search(img, label)
            param_list.append(best_params)
            cluster_scores.append(score)

        best_params = vote_best_params(param_list)
        #print(f"投票完成，最佳路径是{best_params}")
        cluster_params[cluster] = best_params
        all_scores.extend(cluster_scores)

        os.makedirs(f'processed_cluster_{cluster}', exist_ok=True)
        for i, idx in enumerate(cluster_indices):
            proc_img = apply_preprocessing(images[idx], best_params)
            cv2.imwrite(f'processed_cluster_{cluster}/img_{i}.png', proc_img * 255)

    with open('cluster_params_BFS.pkl', 'wb') as f:
        pickle.dump(cluster_params, f)
    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

    return cluster_params, kmeans, all_scores


def predict_and_process(
        img_path: str,
        kmeans_model_path: str = 'kmeans_model.pkl',
        params_path: str = 'cluster_params.pkl'
) -> Tuple[np.ndarray, Dict, float, str]:
    """预测新图像的聚类并应用优化的预处理，返回最佳模板类别"""
    # 加载图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f",
               "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    # 加载模型
    with open(kmeans_model_path, 'rb') as f:
        kmeans = pickle.load(f)
    with open(params_path, 'rb') as f:
        cluster_params = pickle.load(f)


    proc_img_resized = cv2.resize(img, (64, 64))
    features = extract_features(proc_img_resized).reshape(1, -1)
    cluster = kmeans.predict(features)[0]

    params = cluster_params.get(cluster, {'alpha': 1.0, 'rot': 0.0, 'scale': 1.0, 'sharp': 1.0})
    proc_img = apply_preprocessing(img, params)
    proc_img_resized = cv2.resize(proc_img, (64, 64))
    feat = proc_img_resized.flatten().reshape(1, -1)
    feat = scaler.transform(feat)
    distances, indices = knn_model.kneighbors(feat, n_neighbors=1)
    score = np.exp(-distances[0][0])
    pred_index = knn_model.predict(feat)[0]
    predicted_category = le.inverse_transform([pred_index])[0]

    return proc_img, params, score, predicted_category


if __name__ == "__main__":

    data_dir = r"C:\Users\吴\Desktop\code\python\principle\bmp"  # 数据集路径，包含多个类别文件夹
    cluster_params, kmeans, scores = cluster_and_optimize(
        data_dir, n_clusters=5, samples_per_cluster=20
    )

    # 打印结果
    #print(, cluster_params)
    #print(, scores)
    print("搜索完成")
    print(count)
    '''# 预测
    total_score = 0.0
    count = 0
    correct = 0
    all_results = []
    test_img_path = data_dir  # 测试图像路径
    for subdir in Path(data_dir).iterdir():
        if not subdir.is_dir():
            continue
        true_label = subdir.name
        for img_path in subdir.glob("*.png"):
            processed_img, used_params, score, predicted_category = predict_and_process(
                str(img_path)
            )
            total_score += score
            count += 1
            print(f"预测第{count}个，精度：{score}，标签值{true_label},预测{predicted_category}")
            if predicted_category == true_label:
                correct += 1
            all_results.append({
                "image": img_path.name,
                "true": true_label,
                "predicted": predicted_category,
                "score": score
            })
    # 输出统计结果
    avg_score = total_score / count if count > 0 else 0.0
    accuracy = correct / count if count > 0 else 0.0
    print(f"\n共评估图片数: {count}")
    print(f"平均匹配得分: {avg_score:.3f}")
    print(f"分类准确率: {accuracy:.3f}")

    8100轮
    分类准确率: 0.841
    '''
    '''
    processed_img, used_params, score, predicted_category = predict_and_process(
        test_img_path, template_dir
    )
    print("预测使用的参数:", used_params)
    print("预测得分:", score)
    print("预测类别:", predicted_category)
    cv2.imwrite("processed_test.png", processed_img * 255)'''