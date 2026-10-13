# Hướng dẫn chạy HIFA trên Google Colab (GPU)

> Mục tiêu: train các kiến trúc có LLM (A1 FLAG-Trader, B1 HARLF, HIFA) trên GPU
> Colab vì máy dev không có GPU. Phần data + env + baseline tài chính + C1 debate
> (không cần GPU) đã chạy xong ở máy local — xem `TODO_DeCuong.md`.
>
> Áp dụng cho Colab Free (T4 ~15GB) hoặc Colab Pro (A100/L4). Qwen2.5-1.5B +
> LoRA fp16/bf16 chạy thoải mái trên T4.

---

## 0. Chuẩn bị (làm 1 lần)

### 0.1. Đưa code lên nơi Colab đọc được — chọn 1 trong 2 cách

**Cách A — GitHub (khuyến nghị):**
1. Tạo repo riêng (private cũng được) trên GitHub, đẩy toàn bộ thư mục `hifa/` lên.
   ```bash
   # chạy ở máy local, trong d:\LUAN_VAN\LV\hifa
   git init
   git add src configs scripts tests requirements.txt *.md
   git commit -m "HIFA US-market baseline + env + C1"
   git branch -M main
   git remote add origin https://github.com/<username>/hifa.git
   git push -u origin main
   ```
   > Lưu ý: KHÔNG cần đẩy `.venv/`, `data/raw/news` (đã xóa), checkpoints lớn.
   > Thêm `.gitignore` cho `.venv/`, `__pycache__/`, `*.zip`, `data/`, `logs/`.

**Cách B — Google Drive:**
1. Nén `hifa/` (bỏ `.venv`) thành `hifa.zip`.
2. Upload lên Google Drive (vd. thư mục `MyDrive/LUAN_VAN/`).

### 0.2. Bật GPU trên Colab
- Mở notebook mới tại https://colab.research.google.com
- Menu: **Runtime → Change runtime type → Hardware accelerator → GPU → Save**
- Kiểm tra: chạy ô `!nvidia-smi` phải thấy T4/L4/A100.

---

## 1. Notebook mẫu — chép từng ô vào Colab

Có sẵn file `notebooks/hifa_colab.ipynb` trong repo (xem mục 4). Hoặc chép tay các ô dưới.

### Ô 1 — Kiểm tra GPU
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
!nvidia-smi
```

### Ô 2 — Lấy code

**Nếu dùng GitHub (Cách A):**
```python
!git clone https://github.com/<username>/hifa.git
%cd hifa
```

**Nếu dùng Drive (Cách B):**
```python
from google.colab import drive
drive.mount('/content/drive')
!cp "/content/drive/MyDrive/LUAN_VAN/hifa.zip" /content/
!cd /content && unzip -q hifa.zip
%cd /content/hifa
```

### Ô 3 — Cài dependencies
```python
# Torch đã có sẵn trên Colab (bản CUDA). Chỉ cài phần còn thiếu.
!pip install -q transformers peft accelerate bitsandbytes \
    gymnasium stable-baselines3 yfinance pandas-ta quantstats \
    pyyaml scipy tensorboard pyarrow
```
> Không cài lại `torch` (Colab đã có bản CUDA đúng). Cài đè dễ gãy CUDA.

### Ô 4 — Lấy dữ liệu Mỹ (fetch lại trên Colab, nhanh ~2 phút)
```python
import sys; sys.path.insert(0, ".")
# Fetch 8 mã GĐ 0 + benchmark (yfinance, không cần key)
!python -m src.data.fetch_ohlcv_us --phase 0 --benchmark both
# Build state dataframe
!python -m src.data.build_dataset --phase 0
```
> Nếu muốn dùng lại đúng data đã fetch ở local, upload `data/processed/phase_0.parquet`
> + `data/raw/ohlcv/*.parquet` qua Drive thay vì fetch lại.

### Ô 5 — Smoke test LLM thật trên GPU (verify trước khi train dài)
```python
!PYTHONPATH=. python scripts/_smoke_test_llm_real.py
```
> Lần đầu sẽ tải Qwen2.5-1.5B (~3GB) — mất 1-3 phút. Trên GPU, forward 1 prompt
> chỉ còn ~0.1-0.3s (so với ~5-7s trên CPU local). Nếu smoke test chạy nhanh và
> không NaN -> GPU đã được dùng đúng.

### Ô 6 — Train A1 (FLAG-Trader-style), 5 seed
```python
# Config đã trỏ sang data Mỹ (configs/phase_a.yaml). Chạy từng seed:
for seed in range(5):
    !PYTHONPATH=. python -m src.training.train_phase_a \
        --config configs/phase_a.yaml --seed {seed}
```
> `PortfolioPolicyLLM` giờ TỰ phát hiện GPU (torch.cuda.is_available()). Không cần
> sửa gì thêm. Checkpoint lưu ở `checkpoints/`. Log ở `logs/` (tensorboard).

### Ô 7 — Theo dõi training (tùy chọn)
```python
%load_ext tensorboard
%tensorboard --logdir logs
```

### Ô 8 — Lưu kết quả về Drive (QUAN TRỌNG — Colab xóa file khi hết phiên)
```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p "/content/drive/MyDrive/LUAN_VAN/hifa_results"
!cp -r checkpoints logs reports "/content/drive/MyDrive/LUAN_VAN/hifa_results/"
print("Đã lưu checkpoints + logs + reports về Drive.")
```

---

## 2. Thứ tự chạy đúng theo đề cương (Giai đoạn 0 → A)

Theo De-Cuong §6.4.2, chạy baseline kiến trúc theo thứ tự ưu tiên:

1. **B1 (HARLF frozen, ưu tiên 1)** — ít rủi ro train nhất:
   ```python
   # (driver chạy ≥5 seed cho B1/A1/C1 — xem mục 3, cần viết thêm nếu chưa có)
   ```
2. **A1 (FLAG-Trader, ưu tiên 2)** — Ô 6 ở trên.
3. **C1 (Debate)** — chạy với LLM thật:
   ```python
   # C1 không cần train; chỉ cần model để sinh score. Chạy backtest trên test split.
   ```
4. **4 baseline tài chính** — đã chạy ở local (`reports/phase_0/`), hoặc chạy lại:
   ```python
   !PYTHONPATH=. python scripts/_run_baselines_us_phase0.py
   ```

---

## 3. Mẹo tiết kiệm thời gian / tránh mất kết nối Colab

- **Colab Free timeout ~90 phút nếu idle, tối đa ~12h/phiên.** Train nhiều seed có
  thể vượt giới hạn — chia nhỏ: chạy 1-2 seed mỗi phiên, lưu checkpoint về Drive,
  phiên sau load tiếp.
- **Lưu checkpoint thường xuyên về Drive** (Ô 8) ngay sau mỗi seed, phòng mất phiên.
- **Giảm `total_updates`** trong `configs/phase_a.yaml` để chạy thử nhanh trước khi
  cam kết chạy đủ (vd. `total_updates: 20` để kiểm pipeline, rồi mới đặt lại 200).
- **Dùng `--smoke-test`** để verify pipeline cực nhanh trước khi train thật:
  ```python
  !PYTHONPATH=. python -m src.training.train_phase_a --config configs/phase_a.yaml --smoke-test
  ```
- **bf16 vs fp16:** mặc định code dùng `torch.bfloat16`. T4 KHÔNG hỗ trợ bf16 tốt
  (chậm) — trên T4 nên đổi sang fp16. Xem mục 5.

---

## 4. Notebook sẵn sàng

File `notebooks/hifa_colab.ipynb` trong repo đã gộp sẵn các ô trên. Trên Colab:
**File → Upload notebook** (hoặc mở trực tiếp từ GitHub: File → Open notebook → GitHub tab).

---

## 5. Điểm cần lưu ý về dtype (T4 vs A100/L4)

`PortfolioPolicyLLM(torch_dtype=...)` mặc định `torch.bfloat16`:
- **A100 / L4 (Colab Pro):** giữ `bfloat16` — tốt.
- **T4 (Colab Free):** T4 không có đơn vị bf16 phần cứng → chậm. Nên đổi sang `float16`.
  Cách đổi nhanh: sửa `configs/phase_a.yaml` thêm khóa, hoặc sửa default trong code.
  Hiện code chưa đọc dtype từ config cho train_phase_a — nếu cần, báo mình thêm.

---

## 6. Checklist chạy trên Colab (tick khi xong)

- [ ] Bật GPU runtime, `nvidia-smi` thấy GPU
- [ ] Clone/giải nén code, `%cd hifa`
- [ ] Cài deps (Ô 3), KHÔNG cài lại torch
- [ ] Fetch + build data Mỹ (Ô 4) — hoặc upload từ Drive
- [ ] Smoke test LLM thật chạy nhanh, không NaN (Ô 5)
- [ ] Train A1 ≥5 seed (Ô 6)
- [ ] Lưu checkpoints + logs + reports về Drive (Ô 8)
- [ ] Tải reports về máy local để viết luận văn
