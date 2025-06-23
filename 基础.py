from __future__ import annotations
import cv2, numpy as np, time, json, importlib, traceback
from pathlib import Path
from datetime import datetime

CURRENT_IMG_PATH: str | None = None   
LAST_OPT_PARAMS: dict | None = None   
LAST_OPT_TAG:    str  | None = None   

# ===== 基本常量 =====
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TEMPLATE_DIR = Path(__file__).parent / "templates"

def _auto_pick(algo_dict: dict, default_key: int = 0):
    """
    若字典里只有一个算法，或想固定默认，就直接返回对应函数；
    default_key=0 表示取字典里 key==0 的条目。
    """
    return algo_dict[default_key][1]    # 返回对应函数对象


# ---------- 阶段 1 · 车牌检测 ----------
def detect_plate_basic(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h) if h else 0
            area = w * h
            if 2 < ratio < 6 and area > best_area:
                best, best_area = (x, y, w, h), area
    return best
plate_detection_algos = {0: ("基准检测", detect_plate_basic)}

# ---------- 工具：二值化 ----------
def ensure_plate_binary(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bin_img

# ---------- 阶段 2 · 基准切割 ----------
# ---------- NEW：按切割线数组分割 ----------
def segment_chars_by_cuts(img: np.ndarray, cuts: list[int]):
    bin_img = ensure_plate_binary(img)
    chars = []
    for i in range(len(cuts) - 1):
        sub = bin_img[:, cuts[i]:cuts[i + 1]]
        proj = (sub // 255).sum(axis=1)
        thr  = proj.max() * 0.05
        y_top = next((j for j, v in enumerate(proj) if v > thr), 0)
        y_bot = next((j for j in range(len(proj) - 1, -1, -1) if proj[j] > thr),
                     len(proj) - 1)
        chars.append(sub[y_top:y_bot + 1, :])
    return chars

def segment_chars_bin_param(img: np.ndarray, *, K: int = 7, proj_ratio: float = 0.05):
    bin_img = ensure_plate_binary(img)
    H, W   = bin_img.shape
    unit   = max(1, W // K)
    chars  = []
    for i in range(K):
        x1, x2 = i*unit, (W if i==K-1 else (i+1)*unit)
        sub    = bin_img[:, x1:x2]
        proj   = (sub//255).sum(axis=1)
        thr    = proj.max() * proj_ratio
        y_top  = next((j for j,v in enumerate(proj) if v>thr), 0)
        y_bot  = next((j for j in range(len(proj)-1,-1,-1) if proj[j]>thr), len(proj)-1)
        chars.append(sub[y_top:y_bot+1, :])
    return chars

def _make_wrapper(mod_name: str, tag: str):
    def _seg(img: np.ndarray, *, K=6, proj_ratio=0.05):
        global LAST_OPT_PARAMS, LAST_OPT_TAG
        img_path = CURRENT_IMG_PATH or "./demo.jpg"
        try:
            mod  = importlib.import_module(mod_name)
            best = mod.search_best(img_path=img_path) or {}
            if "cuts" in best:
                cuts = list(map(int, best["cuts"]))
                LAST_OPT_PARAMS = {"img_path": img_path, "cuts": cuts}
                LAST_OPT_TAG    = tag
                return segment_chars_by_cuts(img, cuts)
            
            K          = int(best.get("K", K))
            proj_ratio = float(best.get("proj_ratio", proj_ratio))
            print(f"[{tag}] Best Parameters → K={K}, proj_ratio={proj_ratio:.4f}")
            LAST_OPT_PARAMS = {"img_path": img_path, "K": K, "proj_ratio": proj_ratio}
            LAST_OPT_TAG    = tag
            return segment_chars_bin_param(img, K=K, proj_ratio=proj_ratio)
        except Exception:
            print(f"[WARN] Use {mod_name}.search_best() Fail，Use base parameters。")
            traceback.print_exc(limit=1)
            LAST_OPT_PARAMS = {"img_path": img_path, "K": K, "proj_ratio": proj_ratio}
            LAST_OPT_TAG    = tag
            return segment_chars_bin_param(img, K=K, proj_ratio=proj_ratio)
    return _seg


segment_hc  = _make_wrapper("optimize_hc",  "HC")
segment_gd  = _make_wrapper("optimize_gd",  "GD")
segment_ts  = _make_wrapper("optimize_ts",  "TS")
segment_hs  = _make_wrapper("optimize_hs",  "HS")
segment_aco = _make_wrapper("optimize_aco", "ACO")
segment_pso = _make_wrapper("optimize_pso", "PSO")
segment_ga  = _make_wrapper("optimize_ga",  "GA")
segment_sa  = _make_wrapper("optimize_sa",  "SA")

segmentation_algos = {
    0: ("Base", segment_chars_bin_param),
    1: ("HC Superior",     segment_hc),
    2: ("GD Superior",     segment_gd),
    3: ("TS Superior",     segment_ts),
    4: ("HS Superior",     segment_hs),
    5: ("ACO Superior",    segment_aco),
    6: ("PSO Superior",    segment_pso),
    7: ("GA Superior",     segment_ga),
    8: ("SA Superior",     segment_sa),
}

# ---------- 阶段 3 · 字符识别 ----------
def recognize_one_char_basic(char_img: np.ndarray):
    best_ch, best_score = None, float("-inf")
    for ch in ALPHABET:
        p = TEMPLATE_DIR / f"{ch}.png"
        if not p.exists(): continue
        tmpl = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if tmpl is None: continue
        resized = cv2.resize(char_img, tmpl.shape[::-1])
        _, bin_resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        score = cv2.matchTemplate(bin_resized, tmpl, cv2.TM_CCOEFF_NORMED).max()
        if score > best_score:
            best_ch, best_score = ch, score
    return best_ch or "?", best_score if best_ch else 0.0
recognition_algos = {0: ("基准识别", recognize_one_char_basic)}

# ---------- 交互工具 ----------
def choose_model(stage: str, algos: dict):
    print(f"\n--- For {stage} choose model ---")
    for k, (name, _) in algos.items():
        print(f" [{k}] {name}")
    while True:
        try:
            sel = int(input("Enter number: "))
            if sel in algos:
                print(f"[INFO] {stage} use → {algos[sel][0]}")
                return algos[sel][1]
        except Exception:
            pass
        print("None, pls retry …")

# ---------- 评估（供优化脚本调用，也给 main() 写 JSON） ----------
def evaluate(params: dict,
             *, return_detail=False,
             save_json: str | Path | None = None,
             extra: dict | None = None):
    img_path = params.get("img_path", "./demo.jpg")
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)

    t0 = time.time()
    box = detect_plate_basic(img)
    detect_time = time.time() - t0
    if box is None:
        return {"score":0.0, "error":"plate_not_found"} if return_detail else 0.0
    x,y,w,h = box; plate = img[y:y+h, x:x+w]

    t0 = time.time()
    if "cuts" in params:
        chars = segment_chars_by_cuts(plate, list(map(int, params["cuts"])))
    else:
        chars = segment_chars_bin_param(
            plate,
            K=int(params.get("K", 6)),
            proj_ratio=float(params.get("proj_ratio", 0.05)))
    seg_time = time.time() - t0

    t0 = time.time()
    plate_txt, char_scores = "", []
    for c in chars:
        ch, sc = recognize_one_char_basic(c)
        plate_txt += ch; char_scores.append(sc)
    rec_time = time.time() - t0
    score = float(np.mean(char_scores) if char_scores else 0.0)

    t0 = time.time()
    plate_txt, char_scores = "", []
    for c in chars:
        ch, sc = recognize_one_char_basic(c)
        plate_txt += ch
        char_scores.append(float(sc))          
    rec_time = time.time() - t0
    score = float(np.mean(char_scores) if char_scores else 0.0)

    detail = {
        "score"      : score,
        "plate"      : plate_txt,
        "char_scores": char_scores,
        "n_chars"    : len(chars),
        "detect_time": detect_time,
        "seg_time"   : seg_time,
        "rec_time"   : rec_time,
        "params"     : params,
        "timestamp"  : datetime.now().isoformat(),
    }
    if extra:
        detail.update(extra)
    def _np2py(o):
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()        
        if isinstance(o, np.ndarray):
            return o.tolist()      
        raise TypeError
    if save_json:
        p = Path(save_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(detail, ensure_ascii=False, indent=2, default=_np2py))
    return detail if return_detail else score

# ---------- 主程序 ----------
def main():
    global CURRENT_IMG_PATH, LAST_OPT_PARAMS, LAST_OPT_TAG
    CURRENT_IMG_PATH = input("Please enter photo path: ").strip()
    if not Path(CURRENT_IMG_PATH).exists():
        print("[ERROR] image DNE"); return

    detect  = _auto_pick(plate_detection_algos)
    segment = choose_model("Cutting Time", segmentation_algos)
    recog   = _auto_pick(recognition_algos)

    img = cv2.imread(CURRENT_IMG_PATH)
    box = detect(img)
    if box is None:
        print("[ERROR] Cannot find license plate"); return
    x,y,w,h = box; plate = img[y:y+h, x:x+w]

    chars = segment(plate, K=6, proj_ratio=0.05)
    print(f"[INFO] find {len(chars)} cutting picture，show 20 s …")
    for i,c in enumerate(chars):
        cv2.imshow(f"Char_{i+1}", c)
    cv2.waitKey(1); time.sleep(20); cv2.destroyAllWindows()

    try:
        correct = int(input(f"Cutting correctly has（/6）: "))
        correct = max(0, min(6, correct))
    except ValueError:
        correct = 0
    accuracy = correct / 6
    print(f"[INFO] Write down accuracy: {accuracy:.3f}")
    
    # 识别
    plate_txt, scores = "", []
    for c in chars:
        ch, sc = recog(c)
        plate_txt += ch; scores.append(sc)
    print("\n=== 识别结果 ===")
    print("车牌:", plate_txt)
    print("字符置信度:", [f"{s:.3f}" for s in scores])

    tag   = LAST_OPT_TAG or "baseline"
    params= LAST_OPT_PARAMS or {"img_path": CURRENT_IMG_PATH, "K":6, "proj_ratio":0.05}
    out   = Path("results") / f"{tag}_{datetime.now():%Y%m%d-%H%M%S}.json"
    evaluate(params,
             return_detail=True,
             save_json=out,
             extra={"correct_chars": correct, "accuracy": accuracy})   # ★ 传 extra
    print(f"[INFO] Safed → {out}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
