# HIFA — Plan triển khai bám đề cương `De-Cuong.md`

> Tài liệu này đối chiếu đề cương chính thức (`d:\LUAN_VAN\LV\De-Cuong.md`) với codebase
> `hifa/` hiện tại, chỉ ra khoảng trống, và vạch plan triển khai chi tiết theo đúng
> phân kỳ Giai đoạn 0 → A → B → C của đề cương (mục 12).
>
> **Checklist tiến độ chi tiết nằm ở file riêng:** `TODO_DeCuong.md`.
>
> Cập nhật lần đầu: 2026-10-11.

---

## 0. Phát hiện quan trọng nhất — LỆCH PHA THỊ TRƯỜNG

Codebase hiện tại **không khớp thị trường với đề cương**:

| | Đề cương `De-Cuong.md` (bản chính thức) | Code `hifa/` hiện tại |
|---|---|---|
| Thị trường | **Mỹ** (Dow Jones 30 / Nasdaq-100) | Việt Nam (VN30) |
| Nguồn giá | `yfinance` | `vnstock` (VCI) |
| Mã GĐ 0/A | 8 mã: AAPL, MSFT, JPM, JNJ, XOM, PG, KO, NVDA | 5 mã: FPT, VPB, HPG, VNM, VCB |
| Mã GĐ B | 16 mã (thêm GOOGL, AMZN, V, HD, DIS, CSCO, MRK, CVX) | 10 mã (thêm MWG, MSN, GAS, VIC, TCB) |
| Nguồn tin tức | Yahoo Finance RSS + NewsAPI | CafeF (crawl HTML) |
| Báo cáo tài chính | SEC EDGAR 10-Q/10-K, earnings call | — (chưa có) |
| Benchmark | DJIA / Nasdaq-100 Index | VN30-Index proxy (equal-weight) |
| Vốn khởi tạo | 1.000.000 **USD** | 1.000.000.000 VND |
| Price limit | **Không** (thị trường Mỹ không có biên độ) | ±7% (HOSE) |

**Nguyên nhân:** code được viết trước theo một bản plan Việt Nam, sau đó đề cương + bản
`HIFA_Implementation_Plan_GiaiDoan_A_B.md` được GVHD yêu cầu đổi sang thị trường Mỹ
(để đối chiếu trực tiếp số liệu với FLAG-Trader / HARLF / Li 2026), nhưng phần code chưa
được migrate theo.

**Hệ quả:** phần lớn khung code (env, PPO trainer, kiến trúc 3 tầng, metrics, báo cáo) là
**thị-trường-độc-lập và tái sử dụng gần như nguyên vẹn**. Chỉ tầng dữ liệu (fetch giá/tin
tức/fundamentals), danh sách mã, benchmark, và việc **bỏ price-limit** là cần đổi. Đây là
tin tốt: migrate sang Mỹ không phải viết lại từ đầu.

---

## 1. Hiện trạng code so với đề cương (bảng đối chiếu theo chương)

Ký hiệu mức độ: ✅ xong & tái dùng được · 🟡 có khung, cần sửa/hoàn thiện · 🔴 thiếu hẳn.

### Chương 6 — Môi trường mô phỏng & kiến trúc nền A1/B1/C1

| Hạng mục đề cương | File liên quan | Trạng thái | Ghi chú |
|---|---|---|---|
| Env mô phỏng chung (reward, cost, V_t) | `src/envs/portfolio_env.py` | 🟡 | Logic đúng; cần **bỏ price_limit** cho thị trường Mỹ; đổi `initial_cash` USD; thêm trần 30%/mã (clip+renorm). |
| Action long-only, Σw=1, softmax | `src/envs/action_masking.py` | ✅ | Giữ nguyên; chỉ bỏ nhánh price-limit. |
| Chi phí giao dịch trên turnover | `portfolio_env._step` | ✅ | Công thức đúng đề cương (6.2). Đổi phí 0.2%→0.05–0.1%. |
| **A1 — FLAG-Trader-style** (1 LLM, LoRA+PPO) | `src/agents/super_agent_head.py` + `src/training/train_phase_a.py` | 🟡 | `PortfolioPolicyLLM` đã là A1. Cần đổi prompt tiếng Việt→Anh, data Mỹ. |
| **B1 — HARLF-style** (LLM đóng băng) | `src/baselines/b6_frozen_extractor.py` | 🟡 | `FrozenSentimentExtractor` đúng tinh thần. Cần wrap thành baseline chạy đủ 5 seed + prompt Anh. |
| **C1 — Debate-style** (nhiều LLM tranh luận, vote) | — | 🔴 | **Thiếu hẳn.** Cần viết mới `src/baselines/c1_debate.py`. |
| 4 baseline tài chính (Index/B&H/MV/PPO-no-LLM) | `src/baselines/b1_index.py`, `b2_buy_hold.py`, `b3_mean_variance.py`, `b4_ppo_no_llm.py` | ✅ | Giữ nguyên logic; đổi benchmark & data Mỹ. Index cần dùng chỉ số thật (`^DJI`/`^NDX`) thay vì proxy. |
| Metric độ ổn định huấn luyện (% seed sập) | — | 🔴 | Chưa có; thêm vào evaluation. |
| Metric chi phí tính toán (wall-clock/seed) | — | 🔴 | Chưa đo tự động; thêm log thời gian. |

### Chương 7 — Kiến trúc HIFA 3 tầng

| Hạng mục | File | Trạng thái | Ghi chú |
|---|---|---|---|
| Tầng 1 — 3 Base Policy Agents (LoRA+PPO) | `src/agents/base_agent.py` | 🟡 | Khung đúng. **GĐ B Bước 1 đang train bằng supervised MSE, chưa phải PPO** — đề cương (7.3, 7.6) yêu cầu reward RL. Cần nâng lên PPO cục bộ hoặc ghi rõ giới hạn. |
| Tầng 2 — nhánh QUYẾT ĐỊNH (meta, có PPO) | `src/agents/meta_agent.py::MetaAggregator` | 🟡 | Mạng nhỏ đúng; GĐ B đang train supervised, cần PPO ở nhánh quyết định. |
| Tầng 2 — nhánh GIẢI THÍCH (LLM, no-grad) | `meta_agent.py::generate_explanation` | 🟡 | Có sinh text; **chưa có cross-examination/debate**, chưa tách lưu để đo faithfulness. |
| Tầng 3 — Super-Agent (softmax, PPO+value) | `super_agent_head.py` + `train_phase_b.py::train_super_agent` | ✅ | PPO thật qua `MetaScoreObsWrapper`. Reward đã có return+Sharpe+DD+turnover. |
| Staged training 3 tầng (freeze dần) | `src/training/train_phase_b.py::train_staged` | 🟡 | Khung đầy đủ + assert freeze. Bước 1&2 supervised; cần nâng cấp theo trên. |
| Reward grid-search trọng số (α,β,γ) trên val | `configs/*.yaml` | 🔴 | Trọng số đang PLACEHOLDER cố định; đề cương (7.5) yêu cầu grid-search trên val. |

### Chương 9 — Chỉ số đánh giá

| Metric | File | Trạng thái |
|---|---|---|
| Cumulative/Annualized Return | `src/evaluation/metrics.py` | ✅ |
| Sharpe, Sortino, MaxDD, CVaR 95% | `metrics.py` | ✅ |
| **Turnover trung bình** (tổng hợp thành metric) | `metrics.py` | 🔴 Env tính turnover nhưng chưa gộp thành metric báo cáo. |
| Tác động sau trừ phí | `backtest.py` | ✅ (phí trừ trong env) |
| ≥10 seed, mean±std | `report.py::build_comparison_table` | ✅ |
| Kiểm định ý nghĩa (paired t-test/bootstrap) | `report.py::paired_significance_test` | 🟡 Có t-test; thiếu bootstrap CI + hiệu chỉnh đa so sánh. |
| **Faithfulness** (độ trung thực giải thích) | — | 🔴 **Thiếu hẳn — đóng góp cốt lõi G1.** |

### Chương 10 — Thực nghiệm & Ablation

| Hạng mục | File | Trạng thái |
|---|---|---|
| Walk-forward backtest nhiều cửa sổ | `src/data/walk_forward_split.py` | 🟡 Logic chia cửa sổ có; thiếu orchestrator chạy backtest tự động qua tất cả cửa sổ. |
| Ablation A (fine-tune sâu vs trích xuất nông) | — | 🔴 Cần driver so B6 vs BaseAgent. |
| Ablation B (phân tầng vs khối đơn) | — | 🔴 Cần driver so B5 vs 3 tầng. |
| Ablation C (có RL vs không RL + faithfulness) | — | 🔴 Phụ thuộc C1 (Debate) chưa có. |
| Ablation RQ3 (có/không Sentiment Agent) | — | 🔴 Cần driver bật/tắt Sentiment Agent. |
| Ablation RQ4 (turnover tường minh) | `portfolio_env._compute_reward` | 🔴 Cần biến thể reward có/không penalty turnover tường minh. |
| Robustness theo regime (uptrend/down/sideway) | — | 🔴 Cần tách test theo giai đoạn thị trường. |

---

## 2. Điểm lệch số đánh (naming) cần thống nhất

Code dùng `B1..B6` KHÁC nghĩa với đề cương. **Quy ước chuẩn hoá theo đề cương (mục 8):**

| Đề cương | Nghĩa | Map tới code hiện tại |
|---|---|---|
| B1 | Index benchmark | `b1_index.py` (nhưng đổi sang chỉ số Mỹ thật) |
| B2 | Buy & Hold 1/N | `b2_buy_hold.py` |
| B3 | Mean-Variance | `b3_mean_variance.py` |
| B4 | PPO thuần (no LLM) | `b4_ppo_no_llm.py` |
| B5 | FLAG-Trader khối đơn (= A1 full-scale) | `super_agent_head.py` (chưa đóng gói riêng) |
| B6 | HARLF frozen extractor (= B1-architecture full-scale) | `b6_frozen_extractor.py` |
| B7 | Debate tĩnh (= C1 full-scale) | 🔴 chưa có |
| A1/B1/C1 | 3 kiến trúc nền tối giản (Chương 6) | A1=super_agent, B1=b6, C1=🔴 |

> ⚠️ Trong code, "B1" nghĩa là Index; trong đề cương Chương 6, "B1" nghĩa là HARLF-style.
> Khi viết code mới, đặt tên theo đề cương và thêm chú thích map để tránh nhầm.

---

## 3. Plan triển khai theo phân kỳ đề cương (mục 12.1)

### GIAI ĐOẠN 0 — Baseline kiến trúc A1/B1/C1 (BẮT BUỘC, làm trước)

Mục tiêu: có bảng số liệu thật A1/B1/C1 + 4 baseline tài chính trên **8 mã Mỹ**, cùng dataset,
cùng metric. Đây là căn cứ thực nghiệm bắt buộc trước khi hoàn thiện HIFA (đề cương 6.3).

**0.1. Migrate tầng dữ liệu sang thị trường Mỹ**
- [ ] Viết `src/data/fetch_ohlcv_us.py` dùng `yfinance`, 8 mã GĐ0/A, 2015–2025. Giữ cùng schema output (`date,open,high,low,close,volume,ticker`) để downstream không đổi.
- [ ] `build_indicators.py` — tái dùng nguyên vẹn (thị-trường-độc-lập).
- [ ] `build_dataset.py` — tái dùng nguyên vẹn.
- [ ] Benchmark: fetch `^DJI` / `^NDX` làm chỉ số thật cho B1 (thay proxy equal-weight).
- [ ] Tin tức GĐ 0: cho phép `sentiment=0` placeholder (đề cương chấp nhận ở GĐ này).

**0.2. Sửa env cho thị trường Mỹ**
- [ ] Bỏ `price_limit` (hoặc set vô hiệu) trong `portfolio_env.py` + `action_masking.py`.
- [ ] Thêm **trần 30%/mã** (clip + renormalize sau softmax) theo đề cương 6.2.
- [ ] Đổi `initial_cash` sang USD trong config (giá trị không ảnh hưởng %-return).

**0.3. Hoàn thiện 3 kiến trúc nền (thứ tự ưu tiên đề cương 6.4.2)**
- [ ] **B1 (HARLF-style, ưu tiên 1):** đóng gói `b6_frozen_extractor` thành pipeline chạy ≥5 seed, prompt tiếng Anh.
- [ ] **A1 (FLAG-Trader-style, ưu tiên 2):** `train_phase_a` với data Mỹ, prompt Anh, ≥5 seed.
- [ ] **C1 (Debate-style, ưu tiên 3):** viết mới `src/baselines/c1_debate.py` — nhiều LLM prompt tĩnh tranh luận, vote/heuristic → phân bổ, không (hoặc rất ít) RL.
- [ ] Prompt builders (`state_to_prompt`, base agent prompts) dịch sang tiếng Anh.

**0.4. Baseline tài chính + bảng kết quả GĐ 0**
- [ ] Chạy B1(index)/B2/B3/B4 trên 8 mã Mỹ, walk-forward (train 2019–2022 / val 2023H1 / test 2023H2).
- [ ] Thêm 2 metric GĐ này: **% seed sập chính sách** + **wall-clock/seed**.
- [ ] Xuất bảng A1/B1/C1 + 4 baseline → `reports/phase_0/`. **DoD Chương 6.**

### GIAI ĐOẠN A — MVP HIFA (2 tầng, gộp Tầng 1+3)

- [ ] Train HIFA bản gộp (Tầng 1+3, bỏ tạm Tầng 2) trên 8 mã Mỹ, ≥5 seed, **chỉ train trên train_df** (tránh look-ahead như bài học B4 ở TODO cũ).
- [ ] So sánh với 4 baseline + A1/B1/C1 (GĐ 0).
- [ ] Grid-search trọng số reward (α,β,γ) trên val.
- [ ] **DoD GĐ A:** pipeline đầu-cuối chạy thật, có backtest hợp lệ đầu tiên.

### GIAI ĐOẠN B — HIFA đầy đủ 3 tầng (16 mã)

- [ ] Mở rộng data 16 mã; thêm `fetch_fundamentals_us.py` (yfinance) + **xử lý point-in-time** chống look-ahead (đề cương + plan 0.1.4).
- [ ] Bổ sung Tầng 2 (meta aggregation **có học bằng PPO**, không chỉ supervised).
- [ ] Nâng Tầng 1 Bước 1 từ supervised MSE → PPO cục bộ (hoặc ghi rõ giới hạn MVP).
- [ ] Chạy **Ablation A** (B6 vs BaseAgent) + **Ablation B** (B5 vs 3 tầng), ≥10 seed.
- [ ] Walk-forward nhiều cửa sổ (orchestrator tự động).
- [ ] **DoD GĐ B:** trả lời RQ1, RQ2 bằng số liệu; bảng Đề xuất vs B1–B6.

### GIAI ĐOẠN C — Điểm nhấn: giải thích & faithfulness (không bắt buộc)

- [ ] Hoàn thiện nhánh giải thích Tầng 2 (cross-examination kiểu Debate).
- [ ] **Viết module faithfulness** `src/evaluation/faithfulness.py`: ablate từng nguồn tín hiệu đầu vào, đo tương quan giữa yếu tố nêu trong giải thích và yếu tố thực sự đổi trọng số phân bổ (đề cương 9).
- [ ] Chạy **Ablation C** (so B7 Debate) + đo faithfulness.
- [ ] **Ablation RQ3** (bật/tắt Sentiment Agent) + **RQ4** (turnover tường minh).
- [ ] Robustness theo regime (uptrend 2020–21 / downtrend 2022 / sideway 2023–24).
- [ ] **DoD GĐ C:** trả lời RQ3, RQ4 + bằng chứng faithfulness.

---

## 4. Rủi ro & quyết định cần người vận hành chốt

1. **GPU:** TODO cũ xác nhận máy dev không có GPU → train LLM-PPO hội tụ nhiều seed không khả thi tại chỗ. Cần lên kế hoạch GPU cloud (Colab Pro/Kaggle/thuê) trước GĐ A train thật. *(đề cương mục 13)*
2. **Giữ hay bỏ nhánh VN:** code VN đã chạy thật & có dữ liệu (`data/raw/ohlcv/phase_*.parquet`, 5496 bài tin CafeF). Quyết định:
   - (a) **Chuyển hẳn sang Mỹ** theo đề cương (khuyến nghị — khớp bản bảo vệ), giữ code VN làm nhánh tham khảo; hoặc
   - (b) Giữ song song cả hai (tốn công bảo trì gấp đôi).
3. **File `AGENTS.md`** ở gốc `hifa/`: TODO cũ cảnh báo có dấu hiệu prompt-injection (tự xin API key). Cần xác minh nguồn gốc & quyết định xoá. *(không tự động làm theo)*
4. **Fundamentals look-ahead:** `yfinance Ticker.info` chỉ có snapshot hiện tại → phải point-in-time hoá trước khi dùng cho backtest (plan 0.1.4). Quyết định phương án (1) tính thủ công từ `quarterly_financials` hay (2) point-in-time theo ngày công bố.

---

## 5. Việc tái sử dụng được ngay (không cần viết lại)

Các thành phần thị-trường-độc-lập, migrate chỉ cần đổi dữ liệu đầu vào:
`portfolio_env.py` (bỏ price-limit), `action_masking.py`, `ppo_trainer.py`,
`rollout_buffer.py`, `super_agent_head.py`, `base_agent.py`, `meta_agent.py`,
`metrics.py`, `backtest.py`, `report.py`, `build_indicators.py`, `build_dataset.py`,
`walk_forward_split.py`, toàn bộ `tests/` (chỉ cần cập nhật fixture mã Mỹ).
