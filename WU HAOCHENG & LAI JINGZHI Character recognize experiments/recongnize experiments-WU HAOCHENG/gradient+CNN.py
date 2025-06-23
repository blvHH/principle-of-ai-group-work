import cv2
import numpy as np
from pathlib import Path
import time
import csv
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.cluster import KMeans

# Path and parameter settings
TEMPLATE_DIR = Path("templates")
ALPHABET = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
BMP_DIR = Path(r"C:\Users\吴\Desktop\code\python\principle\bmp")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ALPHABET  # 36 classes (0-9, A-Z)

# Custom CNN model
class FlexibleCNN(nn.Module):
    def __init__(self, num_classes=36, layers_enabled=None, kernel_sizes=None):
        super(FlexibleCNN, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 5, 3, 3, 3, 3, 3]
        if layers_enabled is None:
            layers_enabled = [1, 1, 1, 1, 1, 1, 1, 1]
        self.layers_enabled = layers_enabled
        self.kernel_sizes = kernel_sizes
        self.channel_plan = [16, 32, 64, 128, 128, 256, 256, 512]
        self.convs = nn.ModuleList()
        in_channels = 1
        for i in range(len(self.channel_plan)):
            if self.layers_enabled[i]:
                conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.channel_plan[i],
                    kernel_size=self.kernel_sizes[i],
                    padding=self.kernel_sizes[i] // 2
                )
                self.convs.append(conv)
                in_channels = self.channel_plan[i]
            else:
                self.convs.append(None)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x
        for i, conv in enumerate(self.convs):
            if conv is not None:
                out = F.relu(conv(out))
                if i % 2 == 1:
                    out = self.pool(out)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Custom BMP dataset
class BMPDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        for cls in self.classes:
            cls_path = self.root_dir / cls
            for img_name in cls_path.glob("*.png"):
                self.images.append((img_name, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, label

# Preprocess image
def preprocess_image(img, target_size=(32, 32), alpha=1.0):
    img_eq = cv2.equalizeHist(img)  # Histogram equalization
    img_blur = cv2.GaussianBlur(img_eq, (3, 3), 0)
    img_contrast = cv2.convertScaleAbs(img_blur, alpha=alpha, beta=0)
    edges = cv2.Canny(img_contrast, 100, 200)
    edge_density = np.mean(edges > 0)
    noise = np.std(img_contrast)
    bs = max(5, int(8 * (noise / 255)))
    bs = bs + (bs % 2) - 1
    cval = max(0, int(1.5 * (noise / 255)))
    bw = cv2.adaptiveThreshold(img_contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, bs, cval)
    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=3)
    bw = cv2.dilate(bw, np.ones((3, 3), np.uint8), iterations=1)
    bw = cv2.erode(bw, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.Canny(bw, 10, 50)
    bw = cv2.bitwise_or(bw, cv2.bitwise_and(bw, bw, mask=edges))
    ys, xs = np.where(bw > 0)
    if len(xs) and len(ys):
        bw = bw[ys.min():ys.max()+1, xs.min():xs.max()+1]
    out = cv2.resize(bw, target_size, interpolation=cv2.INTER_AREA)
    feat = np.array([[np.mean(out), np.std(out), edge_density]])
    cluster_id = int(kmeans.predict(feat)[0])
    return out, cluster_id

# Precompute template features
print("Precomputing template features...")
template_kmeans = {}
tmpl_features = []
tmpl_labels = []
for ch in ALPHABET:
    p = TEMPLATE_DIR / f"{ch}.png"
    if not p.exists(): continue
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    m = np.mean(img)
    s = np.std(img)
    edges = cv2.Canny(img, 100, 200)
    d = np.mean(edges > 0)
    tmpl_features.append([m, s, d])
    tmpl_labels.append(ch)
kmeans = KMeans(n_clusters=3, random_state=0).fit(tmpl_features)
for idx, ch in enumerate(tmpl_labels):
    c = int(kmeans.labels_[idx])
    template_kmeans.setdefault(c, []).append(ch)
print("Template clustering done.")

# ORB template data
orb = cv2.ORB_create(500)
tmpl_data = {}
for ch in ALPHABET:
    p = TEMPLATE_DIR / f"{ch}.png"
    if not p.exists(): continue
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    proc = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    kp, des = orb.detectAndCompute(proc, None)
    tmpl_data[ch] = (proc, kp, des)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load pretrained model
model = FlexibleCNN(num_classes=36)
model.load_state_dict(torch.load(r'C:\Users\吴\Desktop\code\python\principle\final_cnn_best_100.pth', map_location=device))
model.to(device)
for param in model.convs[:4]:
    if param is not None:
        param.requires_grad = False

# Recognize a single character
def recognize_one_char(char_img, true_label):
    start = time.time()
    proc, cluster_id = preprocess_image(char_img)
    candidates = template_kmeans.get(cluster_id, ALPHABET)

    # CNN prediction
    proc_rgb = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
    proc_pil = Image.fromarray(proc_rgb)
    input_tensor = transform(proc_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        cnn_outputs = model(input_tensor)
        cnn_predicted = torch.argmax(cnn_outputs, dim=1).item()
        cnn_score = torch.softmax(cnn_outputs, dim=1)[0, cnn_predicted].item()

    # ORB fallback
    if cnn_score < 0.8:
        kp2, des2 = orb.detectAndCompute(proc, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        best_ch, best_score = None, 0.0
        if des2 is not None:
            for ch in candidates:
                tmpl_proc, kp1, des1 = tmpl_data[ch]
                if des1 is None: continue
                matches = bf.match(des2, des1)
                score = len(matches) / max(len(kp1), len(kp2))
                if score > best_score:
                    best_score, best_ch = score, ch
        if best_score < 0.2:
            for ch in ALPHABET:
                tmpl_proc, _, _ = tmpl_data.get(ch, (None, None, None))
                if tmpl_proc is None: continue
                res = cv2.matchTemplate(proc, tmpl_proc, cv2.TM_CCOEFF_NORMED)
                sc = float(res.max())
                if sc > best_score:
                    best_score, best_ch = sc, ch
        end = time.time()
        print(f"Time: {end-start:.2f}s | Character: {best_ch} | Score: {best_score:.3f} (ORB)")
        return best_ch, best_score
    else:
        end = time.time()
        print(f"Time: {end-start:.2f}s | Character: {classes[cnn_predicted]} | Score: {cnn_score:.3f} (CNN)")
        return classes[cnn_predicted], cnn_score

# Main program
if __name__ == '__main__':
    # Fine-tune the model (optional)
    dataset = BMPDataset(root_dir=BMP_DIR, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')
    torch.save(model.state_dict(), 'fine_tuned_cnn.pth')

    # Recognize and save results
    results = []
    if BMP_DIR.exists():
        for sub in BMP_DIR.iterdir():
            if not sub.is_dir(): continue
            label = sub.name
            for img_path in sub.glob('*.png'):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                pred, score = recognize_one_char(img, label)
                results.append({'image': img_path.name, 'true': label, 'pred': pred, 'score': score})
    with open('results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'true', 'pred', 'score'])
        writer.writeheader()
        writer.writerows(results)
    print('Results saved to results.csv')