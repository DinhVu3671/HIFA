# HIFA — TODO / Checklist tiến độ (bám `De-Cuong.md`)

> Checklist này bám **đề cương chính thức** `d:\LUAN_VAN\LV\De-Cuong.md` (thị trường **Mỹ**).
> Phân tích & plan chi tiết: xem `PLAN_DeCuong.md` cùng thư mục.
> File `TODO.md` cũ bám bản plan Việt Nam — giữ lại làm tham khảo, KHÔNG dùng để chấm tiến độ bản bảo vệ.
>
> Trạng thái: `[ ]` chưa làm · `[~]` đang làm/một phần · `[x]` xong · `[!]` bị chặn / cần quyết định
>
> Cập nhật lần cuối: 2026-10-11 (khởi tạo).

---

## ⚠️ 0. QUYẾT ĐỊNH CẦN CHỐT TRƯỚC (blocking)

- [x] **Thị trường:** ĐÃ CHỐT chuyển hẳn sang Mỹ (yfinance/Dow30/Nasdaq-100). Đã xóa toàn bộ data/checkpoint/report VN cũ (phase_a/b parquet, news CafeF, checkpoints phase_a, reports/phase_a) — 2026-10-11. Chỉ còn data Mỹ (phase_0 + benchmark).
  - **Đã dọn nốt CODE VN:** xóa `src/data/fetch_ohlcv.py` (vnstock), `src/data/fetch_news.py` (CafeF), 4 script VN (`_run_baselines_real`, `_run_baselines_walkforward`, `_crosscheck_prices_vndirect`, `_extend_news_crawl`), 2 shell script (`run_phase_a/b.sh`), `tests/test_fetch_news.py`, fixture CafeF.
  - Nội tuyến `get_sentiment_placeholder` vào `build_dataset.py`; `train_phase_b.py` dùng placeholder tin tức (chờ pipeline Yahoo RSS Mỹ GĐ B/C).
  - Cập nhật `configs/phase_a.yaml` + `phase_b.yaml` sang mã Mỹ + env Mỹ (price_limit=null, max_weight=0.30, cash USD). `requirements.txt`: bỏ vnstock, thêm yfinance.
  - **Toàn bộ test pass: 107/107** sau khi dọn.
- [~] **GPU:** dùng Google Colab. ĐÃ CHUẨN BỊ: `COLAB_GUIDE.md` (hướng dẫn chi tiết) + `notebooks/hifa_colab.ipynb` (notebook sẵn chạy) + `.gitignore`. `PortfolioPolicyLLM` đã thêm auto-detect GPU (`device=None` → cuda nếu có). CẦN: đẩy code lên GitHub/Drive + chạy notebook trên Colab GPU.
- [!] **`AGENTS.md`** ở gốc repo nghi prompt-injection → xác minh & quyết định xoá.
- [ ] **Fundamentals point-in-time:** chọn phương án (1) tính từ `quarterly_financials` hay (2) theo ngày công bố (chống look-ahead) — chốt trước GĐ B.
- [ ] **Phí giao dịch:** chốt 0.05% hay 0.1%/lệnh (đề cương cho dải 0.05–0.1%).
- [ ] **Trọng số reward (α,β,γ):** xác nhận sẽ grid-search trên val, không gán cố định.

---

## 1. GIAI ĐOẠN 0 — Baseline kiến trúc A1/B1/C1 (BẮT BUỘC, làm trước HIFA)

### 1.1 Migrate tầng dữ liệu sang Mỹ (yfinance)
- [x] `src/data/fetch_ohlcv_us.py` — yfinance, 8 mã (AAPL,MSFT,JPM,JNJ,XOM,PG,KO,NVDA), 2015–2025, schema giữ nguyên ✅ *(verify AAPL + fetch rổ 8 mã = 22120 dòng, auto-adjust giá)*
- [x] Fetch benchmark thật `^DJI` / `^NDX` cho B1 ✅ *(2765 dòng mỗi index → `data/raw/ohlcv/benchmark_{dow,nasdaq100}.parquet`)*
- [x] `build_indicators.py` chạy trên data Mỹ (tái dùng) ✅ *(qua build_dataset, 0 NaN)*
- [x] `build_dataset.py` → `data/processed/phase_0.parquet` (8 mã Mỹ) ✅ *(2732 ngày × 48 cột, 2015-02-20→2025-12-30, 0 NaN)*
- [x] Tin tức GĐ 0: dùng `sentiment=0` placeholder ✅ *(get_sentiment_placeholder, tái dùng)*
- [x] `tests/test_env_us.py` — 6 test cho position cap + price_limit=None + PortfolioEnvUS, pass ✅
- [ ] Cập nhật `tests/` fixtures sang mã Mỹ cho fetch_ohlcv_us (tests hiện pass với fixture tự tạo)

### 1.2 Sửa env cho thị trường Mỹ
- [x] Vô hiệu `price_limit` (hỗ trợ `price_limit=None`) trong `portfolio_env.py` + `action_masking.py` ✅
- [x] Thêm trần **30%/mã** (`apply_position_cap`, clip + renormalize) — đề cương 6.2 ✅ *(verify: mã 0.9→0.30, max per-asset trong episode = 0.3000)*
- [x] `initial_cash` → USD; thêm lớp `PortfolioEnvUS` (price_limit=None, max_weight=0.30, cash=1tr USD, phí 0.05%) ✅
- [x] `tests/test_env.py` + `test_baselines.py` vẫn pass (21/21) sau thay đổi ✅

### 1.3 Config GĐ 0
- [x] `configs/phase_0.yaml` — 8 mã Mỹ, walk-forward train 2019–2022 / val 2023H1 / test 2023H2, phí 0.05%, trần 30%/mã, ≥5 seed ✅

### 1.4 Prompt tiếng Anh
- [x] Dịch `state_to_prompt` (super_agent_head) sang tiếng Anh ✅ (price/RSI/MACD bullish-bearish/volatility)
- [x] Dịch base agent prompts (fundamental/technical/sentiment) sang tiếng Anh ✅
- [x] Dịch prompt `generate_explanation` (meta_agent) sang tiếng Anh ✅
- [x] Cập nhật test prompt (test_phase_b_agents, test_ppo_trainer) sang EN + mã Mỹ ✅

### 1.5 Ba kiến trúc nền (thứ tự ưu tiên đề cương 6.4.2)
- [~] **B1 (HARLF-style, ưu tiên 1)** — `b6_frozen_extractor` sẵn (đóng băng hoàn toàn); CẦN GPU để chạy ≥5 seed trên data Mỹ
- [~] **A1 (FLAG-Trader-style, ưu tiên 2)** — `train_phase_a` + prompt EN sẵn; CẦN GPU để chạy ≥5 seed
- [x] **C1 (Debate-style, ưu tiên 3)** — ĐÃ VIẾT `src/baselines/c1_debate.py` ✅
  - `DebateAllocator` (debate round + vote heuristic, không RL) + `LLMDebateScorer` (LLM đóng băng, tái dùng B6) + `run_c1_debate` (backtest đầy đủ, metric)
  - Chạy được KHÔNG cần GPU khi backtest với scorer giả; chạy LLM thật cần model (tải Qwen)
- [x] `tests/test_c1_debate.py` — 5 test (debate, cap, all-cash, backtest, require-scorer), pass ✅

### 1.6 Baseline tài chính
- [~] B1(index ^DJI thật)/B2/B3/B4 chạy trên 8 mã Mỹ, walk-forward — B1/B2/B3 xong (test 2023H2); B4 đang train (chỉ train trên train_df). Driver: `scripts/_run_baselines_us_phase0.py`
  - B1 (^DJI): Sharpe 1.89, cum 9.5% · B2 (B&H 1/N): Sharpe 0.88 · B3 (MV): Sharpe -1.35 (overfit train)
  - `train_b4` đã thêm hỗ trợ `env_cls`/`env_kwargs` để dùng `PortfolioEnvUS`
- [x] Metric mới: **% seed sập chính sách** — `is_policy_collapsed` + `policy_collapse_rate` trong metrics.py ✅
- [~] Metric mới: **wall-clock/seed** — driver đã đo thời gian train B4; cần gói thành cột báo cáo chuẩn khi chạy đủ seed
- [ ] Chạy đủ ≥5 seed cho B4 (hiện 1 seed) — nhanh trên CPU, có thể làm không cần GPU

### 1.7 Definition of Done — Chương 6
- [ ] Bảng số liệu A1/B1/C1 + 4 baseline tài chính, cùng dataset, ≥5 seed, mean±std → `reports/phase_0/`
- [ ] Nhận xét bằng số liệu hạn chế quan sát được của từng kiến trúc nền
- [ ] Kiểm định không có look-ahead (sanity: Sharpe<3, return<100%/năm)

---

## 2. GIAI ĐOẠN A — MVP HIFA (gộp Tầng 1+3)

- [ ] Train HIFA bản gộp (Tầng 1+3, tạm bỏ Tầng 2) trên 8 mã Mỹ, **chỉ train trên train_df**
- [ ] ≥5 seed, log reward lên wandb/tensorboard
- [ ] Grid-search trọng số reward (α,β,γ) trên val
- [ ] So sánh HIFA-MVP vs 4 baseline + A1/B1/C1 (GĐ 0) → `reports/phase_a/`
- [ ] Evaluation thật trên kết quả training thật
- [ ] **DoD GĐ A:** pipeline đầu-cuối chạy thật + backtest hợp lệ đầu tiên

---

## 3. GIAI ĐOẠN B — HIFA đầy đủ 3 tầng (16 mã)

### 3.1 Mở rộng dữ liệu
- [ ] 16 mã (thêm GOOGL,AMZN,V,HD,DIS,CSCO,MRK,CVX) → `data/processed/phase_b.parquet`
- [ ] Tin tức thật (Yahoo Finance RSS → NewsAPI bổ sung), gắn nhãn theo mã
- [ ] `src/data/fetch_fundamentals_us.py` (yfinance) + **point-in-time** chống look-ahead
- [ ] Walk-forward nhiều cửa sổ (train 3y/val 6m/test 6m, trượt 6m, 2019–2025)

### 3.2 Tầng 1 — 3 Base Agent (RL)
- [ ] Nâng Bước 1 từ supervised MSE → **PPO cục bộ** (hoặc ghi rõ giới hạn MVP)
- [ ] Train độc lập 3 base agent trên reward cục bộ, data 16 mã + tin tức thật

### 3.3 Tầng 2 — Meta Aggregation (nhánh quyết định có học)
- [ ] Train MetaAggregator **bằng PPO** (nhánh quyết định), Tầng 1 freeze (assert `requires_grad=False`)

### 3.4 Tầng 3 — Super-Agent
- [ ] Train Super-Agent PPO đầy đủ, Tầng 1+2 freeze
- [ ] Lưu checkpoint riêng từng tầng, load lại demo được

### 3.5 Baseline đầy đủ & Ablation
- [ ] B5 (FLAG-Trader khối đơn) chạy đầy đủ 16 mã
- [ ] B6 (HARLF frozen) chạy đầy đủ 16 mã
- [ ] **Ablation A** (B6 vs BaseAgent fine-tune) ≥10 seed + t-test
- [ ] **Ablation B** (B5 khối đơn vs 3 tầng) ≥10 seed, so std Sharpe
- [ ] Orchestrator walk-forward tự động chạy qua mọi cửa sổ

### 3.6 Metric còn thiếu
- [x] Thêm **turnover trung bình** (`avg_turnover`) vào `compute_all_metrics` (tham số `turnover_history`) ✅
- [ ] Thêm **bootstrap CI** + hiệu chỉnh đa so sánh (Holm/Bonferroni) vào `report.py`

### 3.7 Definition of Done — GĐ B
- [ ] Bảng đầy đủ Đề xuất vs B1–B6, 16 mã, ≥10 seed, walk-forward
- [ ] **Trả lời RQ1, RQ2** bằng số liệu cụ thể + kiểm định thống kê

---

## 4. GIAI ĐOẠN C — Giải thích & Faithfulness (điểm nhấn, không bắt buộc)

- [ ] B7 (Debate tĩnh) chạy đầy đủ 16 mã
- [ ] Hoàn thiện nhánh giải thích Tầng 2 (cross-examination kiểu Debate), tách lưu text giải thích
- [ ] **`src/evaluation/faithfulness.py`** — ablate từng nguồn tín hiệu, đo tương quan giải thích ↔ yếu tố thực sự đổi trọng số (đề cương 9)
- [ ] **Ablation C** (so B7) + đo faithfulness
- [ ] **Ablation RQ3** — bật/tắt Sentiment Agent (đóng góp thành phần ngôn ngữ)
- [ ] **Ablation RQ4** — biến thể reward có/không penalty turnover tường minh
- [ ] **Robustness theo regime** — tách test uptrend 2020–21 / downtrend 2022 / sideway 2023–24
- [ ] **Trả lời RQ3, RQ4** + bằng chứng faithfulness
- [ ] `tests/` cho faithfulness

---

## 5. Việc viết luận văn / trình bày (song song)
- [ ] Chuẩn hoá naming B1–B7 theo đề cương trong code (hiện lệch — xem PLAN mục 2)
- [ ] Thu thập screenshot log training (wandb) cho luận văn
- [ ] Định dạng tài liệu tham khảo theo chuẩn cơ sở đào tạo (APA/IEEE)

---

## Tổng kết trạng thái (cập nhật thủ công khi tick)

| Giai đoạn | Tiến độ ước lượng | Chặn bởi / việc tiếp theo |
|---|---|---|
| GĐ 0 — Baseline kiến trúc | ~80% — data+env+config Mỹ, prompt EN, C1 debate, metric turnover/collapse, baseline tài chính (1 seed) đều XONG | còn: A1/B1 chạy ≥5 seed (GPU); B4 ≥5 seed (CPU, ~35') |
| GĐ A — MVP | khung sẵn + prompt EN, chưa train Mỹ | GPU |
| GĐ B — 3 tầng | khung sẵn, cần nâng cấp RL Tầng 1/2 + fundamentals Mỹ | GĐ A + GPU |
| GĐ C — Faithfulness | thiếu module faithfulness; C1 đã có, cần B7 (C1 full-scale + Tầng 3) | GĐ B |

**Toàn bộ test: 112/112 pass** (sau khi thêm C1 + dịch prompt EN).
