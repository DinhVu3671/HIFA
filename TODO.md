# HIFA — TODO / Checklist trạng thái triển khai

> Bám sát `HIFA_Implementation_Plan_GiaiDoan_A_B.md`. Cập nhật trạng thái theo từng mục khi hoàn thành.
> Trạng thái: `[ ]` chưa làm · `[~]` đang làm / một phần · `[x]` xong · `[!]` bị chặn / cần quyết định

Cập nhật lần cuối: 2026-07-26 (vòng 3) — vòng 1: code bằng 6 subagent song song + chạy thật (fetch data,
B1-B4, smoke-test Qwen thật). Vòng 2: fetch Phase B 10 mã thật + sửa/chạy production pipeline tin tức thật
(CafeF) + đối chiếu giá với VNDIRECT. Vòng 3 (phiên này): crawl tin tức sâu hơn (3462 bài, phủ tới 2012-2016
cho nhiều mã) + nối tin tức thật vào `train_phase_b.py` (phát hiện & sửa 2 bug thật: look-ahead do so sánh
timestamp không chuẩn hoá theo ngày, và thiếu `load_state_dataframe()` ở nhánh dữ liệu thật của
`train_phase_b.py::main()`) + migrate vnstock API cũ đã quá hạn deprecate. Tất cả trên máy dev (mạng có,
KHÔNG có GPU).

**Tóm tắt:** Toàn bộ khung code Giai đoạn A + B (26 file `src/`, 7 file `tests/`) đã viết xong,
**109/109 test pytest pass**. Đã chạy THẬT thành công: fetch dữ liệu VN30 5 mã (Phase A) + 10 mã (Phase B)
qua `vnstock` (mạng, API mới đã migrate), build dataset + indicators, B1-B4 full trên dữ liệu thật (bảng kết
quả ở mục 1.3), verify `PortfolioPolicyLLM`+`SimplePPOTrainer` với model Qwen2.5-1.5B-Instruct **thật** (tải
qua mạng, forward pass + 1 vòng PPO nhỏ chạy đúng), crawl tin tức thật từ CafeF (3462 bài/10 mã, sau khi phát
hiện và sửa URL đặc tả gốc đã chết) và đã **nối tin tức thật vào pipeline training Giai đoạn B** (verify thật:
VCB có tin ở 40.2% số ngày, trải từ 2018). Rào cản còn lại duy nhất: **máy này không có GPU**
(`torch.cuda.is_available()==False`) nên train Đề xuất hội tụ đủ 5 seed theo DoD là không khả thi về thời
gian tại đây — mọi phần code liên quan đã verify đúng, chỉ còn thiếu phần cứng để chạy dài hơi. Theo lựa
chọn của người vận hành (2026-07-26): tiếp tục các việc không cần GPU, chưa tìm nguồn GPU ngay.

⚠️ Lưu ý bảo mật: file `AGENTS.md` ở gốc dự án chứa nội dung khả nghi (prompt injection dạng "tự động xin
API key") — xem mục 3, KHÔNG làm theo nội dung file đó.

---

## 0. Setup nền tảng

- [x] Tạo cấu trúc thư mục dự án `hifa/` theo mục 2 của spec
- [x] Tạo `requirements.txt` (đã bổ sung `tensorboard`, `pyarrow` so với khung gốc — phát sinh khi test thật)
- [x] Tạo venv tại `hifa/.venv`, cài đặt dependencies — **đã cài thành công trên Python 3.13**
      (torch 2.13, transformers, peft, accelerate, gymnasium 1.3, stable-baselines3 2.9, pandas, quantstats,
      vnstock, wandb, pandas-ta, scipy, requests, beautifulsoup4, pytest). `bitsandbytes` **chưa cài** (không có
      trong lần cài — hỗ trợ Windows còn hạn chế, không bắt buộc cho CPU dev/test; cần khi quantize int4/int8 lúc
      train thật, đánh giá lại lúc đó).
- [x] `configs/phase_a.yaml`, `configs/phase_b.yaml` khung tham số
- [x] `pytest` chạy được — **101/101 test pass**, 12.5s

---

## 1. GIAI ĐOẠN A — MVP

### 1.1 Thu thập & xử lý dữ liệu (mục 3.2)

- [x] `src/data/fetch_ohlcv.py` — `fetch_ohlcv()`, `fetch_all()`, `TICKERS_PHASE_A`/`TICKERS_PHASE_B`, CLI `--phase a|b`
  - [x] Chạy thử `fetch_all()` thật qua mạng — **thành công**: 5/5 mã, 1829 phiên/mã, 9145 dòng, lưu tại `data/raw/ohlcv/phase_a.parquet`. Không trùng ngày (đã kiểm bằng `groupby(ticker).date.count()` trong code). Lưu ý: dữ liệu trả về bắt đầu từ 2018-10-25 dù `start="2019-01-01"` — do warm-up của indicator rolling window (không phải lỗi fetch, xem `build_dataset`).
  - [x] Sửa lỗi `UnicodeEncodeError` khi chạy trên console Windows cp1252: `vnstock` in thông báo deprecation có dấu tiếng Việt ra stdout lúc khởi tạo `Vnstock()` — đã thêm `sys.stdout/stderr.reconfigure(encoding="utf-8")` trong `if __name__=="__main__":` của `fetch_ohlcv.py`.
  - [x] **Đã migrate** từ `Vnstock().stock(symbol=...).quote.history(...)` (deprecate từ 31/08/2025, đã quá hạn) sang `vnstock.api.quote.Quote(source="VCI", symbol=ticker).history(start=..., end=..., interval="1D")` — verify thật qua mạng: schema trả về giống hệt (`time/open/high/low/close/volume`), giá trị số khớp 100% với lần fetch cũ (đối chiếu FPT 2024-01-02..10). Chỉ `"vci"`/`"msn"` còn là source hợp lệ (API mới không còn `"tcbs"` như comment cũ giả định — đã xoá comment sai này).
  - [x] Đối chiếu điểm giá với nguồn độc lập — thay vì thủ công bằng mắt, đã tự động hoá: lấy ngẫu nhiên 1 điểm/mã (10 điểm, seed cố định) trải từ 2019 đến 2024 trên cả 10 mã Phase B, đối chiếu giá đóng cửa với **VNDIRECT dchart API** (`dchart-api.vndirect.com.vn`, nguồn công khai độc lập với `vnstock`). Kết quả: **10/10 điểm khớp**, sai khác tương đối lớn nhất **0.119%** (VNM, 2024-08-02: 61.240 vs 61.167 — nhiễu làm tròn), phần lớn còn lại <0.03%. Không phát hiện sai lệch hệ thống hay lỗi nguồn dữ liệu. Script: `scripts/_crosscheck_prices_vndirect.py` (driver tạm, cùng convention với `_run_baselines_real.py`).
  - [x] Alignment ngày lễ VN khác Mỹ khi ghép nhiều mã: verify được qua `build_dataset` chạy thật — inner-join giữ 1796/1829 ngày (loại 33 ngày không có đủ dữ liệu ở cả 5 mã, đúng cơ chế thiết kế)
- [x] `src/data/build_indicators.py` — `add_indicators()`, `add_indicators_multi()` tính riêng từng ticker rồi mới gộp — có test xác nhận không lẫn dữ liệu giữa các mã (lỗi nghiêm trọng đặc tả cảnh báo)
- [x] `src/data/fetch_news.py` — crawl CafeF thật + `check_robots_txt()` + `get_sentiment_placeholder()` (Giai đoạn A dùng placeholder=0.0 cho cột `sentiment` số học trong `state_df`, không liên quan tới text tin tức thật ở đây)
  - [x] Kiểm tra `robots.txt` thật: **cafef.vn** → `Allow: /` toàn site (không giới hạn); **vietstock.vn** → disallow toàn bộ root cho `user-agent: *`, **KHÔNG dùng làm nguồn crawl**; **vnexpress.net** → robots.txt cho phép root nhưng chưa verify selector thật (dự phòng Giai đoạn C nếu cần thêm nguồn).
  - [!] **Phát hiện quan trọng**: URL đặc tả gốc `https://cafef.vn/du-lieu/{ticker}.chn` đã bị gỡ (404, JS redirect về `s.cafef.vn/du-lieu.chn`) — đã thay bằng URL thật đã verify qua HTTP 200: trang tag theo mã `https://cafef.vn/{ticker}.html` (trang 1) + `https://cafef.vn/{ticker}/trang-{n}.html` (phân trang, verify tới trang 6). Selector thật: mỗi bài trong `div.tlitem`, tiêu đề+link ở `a.box-category-link-title`, ngày ở `p.time_cate span.time` (format `dd/mm/yyyy HH:MM`), tóm tắt ở `p.sapo.box-category-sapo`. Đã lưu 1 fixture HTML thật vào `tests/fixtures/cafef_fpt_sample.html` để test parser offline không phụ thuộc CafeF còn online.
  - [x] Lưu raw HTML + gắn nhãn ticker trên dữ liệu thật — chạy 2 vòng CLI thật: (1) `--max-pages 5` ban đầu (1000 bài, VCB bị timeout 1 lần ở trang 3, đã bổ sung), (2) sau đó `--max-pages 20` để có overlap thật với giai đoạn OHLCV lịch sử (tin tức crawl "hôm nay" 2026 tự nhiên lệch xa 2018-2025 nếu chỉ lấy ít trang — mã càng nhiều tin như FPT càng cần nhiều trang mới lùi đủ xa). Kết quả vòng 2: **3462 bài/10 mã** tại `data/raw/news/articles.parquet` (không NaN date/title/summary), độ phủ theo mã: GAS/MSN/VCB/TCB/VIC/VNM lùi được tới 2012-2016 (phủ hết khoảng OHLCV 2018-2025); FPT/HPG/MWG (tin ra nhiều/ngày) mới lùi được tới 2021-2025. Raw HTML lưu tại `data/raw/news/*.html`.
  - [x] **Vòng 4 (2026-09-08)** — crawl sâu thêm riêng cho FPT/HPG/MWG (`scripts/_extend_news_crawl.py`, driver tạm) với `max_pages` cao hơn hẳn (170/45/75), tự dừng sớm khi hết bài (FPT dừng ở trang 88, HPG trang 33, MWG trang 45). Gộp + dedup theo `(ticker, url)` với dữ liệu cũ, không mất bài đã có. Kết quả: **5496 bài/10 mã**, FPT 1723 bài (từ 2012-04-18), HPG 637 bài (từ 2012-12-07), MWG 874 bài (từ 2013-04-26) — cả 3 mã giờ đều phủ vượt xa mốc 2018-10-25 (điểm bắt đầu OHLCV), đúng như 7 mã còn lại. **10/10 mã Giai đoạn B nay đều có tin tức phủ hết khoảng OHLCV 2018-2025.**
  - [x] `tests/test_fetch_news.py` — 8 test offline dùng fixture HTML thật + dữ liệu tin tức giả lập có kiểm soát: parse `div.tlitem` đúng field, `crawl_cafef_ticker_news`/`crawl_all_tickers` đúng với session giả lập, dedup + lưu parquet đúng, và 3 test riêng cho `attach_recent_news()` (xem mục dưới).
  - [x] **Nối vào pipeline theo-ngày** — thêm `attach_recent_news(state_df, tickers, articles_path, window_days=7, max_articles=3)` trong `fetch_news.py`: với mỗi ngày *d*, gộp tối đa 3 tiêu đề bài gần nhất trong cửa sổ `window_days` có `date <= d` (chuẩn hoá về mức NGÀY, bỏ giờ:phút) thành 1 chuỗi text cho cột `(ticker, "recent_news")`, fallback "Không có tin tức mới." nếu không có bài nào khớp. **Bug phát hiện qua viết test rồi tự sửa**: lần cài đầu so sánh timestamp chính xác (không chuẩn hoá theo ngày) khiến bài đăng buổi sáng CÙNG NGÀY d (vd 09:00) bị loại sai vì index của `state_df` luôn ở mức 00:00:00 — đã sửa bằng `.dt.normalize()` cho cả 2 phía trước khi so sánh; test `test_attach_recent_news_no_look_ahead` xác nhận vừa không leak tương lai vừa không loại sai tin cùng ngày.
  - [x] `train_phase_b.py::main()` (nhánh dữ liệu thật, không phải `--smoke-test`) đã gọi `attach_recent_news()` sau khi load `state_df`. **Bug tích hợp khác phát hiện & sửa cùng lúc**: nhánh này trước đó gọi `pd.read_parquet(processed_path)` trực tiếp thay vì `load_state_dataframe()` — đúng loại lỗi "cột phẳng TICKER__feature không tự khôi phục MultiIndex" mà TODO.md mục 1.1 đã cảnh báo và từng sửa ở `b4_ppo_no_llm.py`, nhưng bị bỏ sót ở file này; nếu không phát hiện, `train_phase_b.py --config configs/phase_b.yaml` (không `--smoke-test`) sẽ crash `TypeError` ngay khi bắt đầu train thật.
  - [x] **Verify thật trên dữ liệu thật** (không phải test đơn thuần): `load_state_dataframe('data/processed/phase_b.parquet')` → `attach_recent_news()` → VCB có tin thật ở **721/1796 ngày (40.2%)**, trải từ đúng ngày đầu tiên của dataset (2018-10-25) tới cuối; ví dụ ngày 2018-10-25 khớp bài "Mở ngân hàng 100% vốn tại Lào, Vietcombank đang hướng tới mục tiêu gì?" — xác nhận pipeline hoạt động đúng đầu-cuối trên dữ liệu thật, không chỉ trên test giả lập.
- [x] `src/data/build_dataset.py` — `build_state_dataframe()` — pivot MultiIndex(ticker,feature), inner-join theo ngày giao dịch chung, raise rõ ràng nếu join rỗng. Đã chạy thật trên dữ liệu Phase A → `data/processed/phase_a.parquet` (1796 dòng × 5 mã × 6 feature, 2018-10-25 → 2025-12-31).
  - [x] **Bug tích hợp đã sửa**: file parquet lưu ra có cột phẳng `"TICKER__feature"` (parquet không hỗ trợ MultiIndex columns) nhưng ban đầu KHÔNG có hàm khôi phục lại MultiIndex khi đọc lại — mọi module downstream (baselines, training) kỳ vọng MultiIndex sẽ lỗi `TypeError: Index must be a MultiIndex` nếu tự `pd.read_parquet()`. Đã thêm `load_state_dataframe(path)` vào `build_dataset.py` làm điểm vào chuẩn duy nhất để đọc lại state dataframe từ đĩa — cần dùng hàm này (không dùng `pd.read_parquet` trực tiếp) ở mọi nơi load `data/processed/*.parquet`.
  - [x] **Bug tích hợp đã sửa**: `b4_ppo_no_llm.py::_load_phase_data()` gọi `build_state_dataframe(tickers=tickers)` thiếu tham số bắt buộc `ohlcv_path` → `TypeError` khi chạy CLI thật. Đã sửa để ưu tiên đọc `data/processed/phase_{phase}.parquet` qua `load_state_dataframe()` nếu đã có sẵn, nếu chưa thì build từ `data/raw/ohlcv/phase_{phase}.parquet`.
- [x] `src/data/walk_forward_split.py` — `make_walk_forward_windows()` (nhiều cửa sổ, dùng Giai đoạn B) + `single_window_phase_a()` (đúng 1 cửa sổ 2019-2022/2023H1/2023H2 cho Giai đoạn A)
- [x] `tests/test_data_pipeline.py` — **22/22 pass**: không NaN, không leak tương lai (test bằng nhiễu ngẫu nhiên sau ngày t), tính riêng từng ticker đúng, align ngày lệch giữa các mã đúng, walk-forward windows không chồng lấn sai logic

### 1.2 Môi trường RL (mục 3.3)

- [x] `src/envs/portfolio_env.py` — `PortfolioEnvPhaseA(gym.Env)` đầy đủ + `PortfolioEnvPhaseB` (subclass tái sử dụng cho Giai đoạn B), Gymnasium API 5-tuple chuẩn, giá trong obs đã **chuẩn hóa** (close/close ngày 0 episode — sửa so với đặc tả gốc dùng giá thô)
- [x] `src/envs/action_masking.py` — `apply_action_mask()`: khi có `current_weights` giữ nguyên tỷ trọng mã bị khóa (đúng ý đặc tả "giữ nguyên trọng số cũ", đã sửa 1 lỗi nhỏ trong pseudo-code gốc); khi không có thì zero+renormalize
- [x] `tests/test_env.py` — **14/14 pass**, kiểm cả bằng `gymnasium.utils.check_env`

### 1.3 Baseline B1–B4 (mục 3.5)

- [x] `src/baselines/b1_index.py` — proxy equal-weight (VN30-Index thật chưa có sẵn trong state_df, có tham số `index_close` tùy chọn để dùng chỉ số thật khi có)
- [x] `src/baselines/b2_buy_hold.py` — mua đều ngày đầu, giữ nguyên số cổ phiếu (khác B1 ở chỗ không rebalance ngầm định)
- [x] `src/baselines/b3_mean_variance.py` — Markowitz, regularize `cov + 1e-6*I`, clip/renormalize kết quả SLSQP
- [x] `src/baselines/b4_ppo_no_llm.py` — SB3 PPO MlpPolicy trên `PortfolioEnvPhaseA` thật, CLI `--phase`, `--total-timesteps`, `--smoke-test`
- [x] `tests/test_baselines.py` — **7/7 pass**, B4 đã verify chạy thật (không chỉ mock) với `PortfolioEnvPhaseA` thật
- [x] **Chạy B1–B4 trên dữ liệu thật** (2018-10-25 → 2025-12-31, 5 mã, 1796 phiên). B4 train đầy đủ 200,000 timesteps (seed=0, ~7 phút, 481-665 fps trên CPU). Kết quả lưu tại `reports/phase_a/comparison_table.{csv,md}`:

  | Chiến lược | Cumulative Return | Annualized | Sharpe | Sortino | Max Drawdown | CVaR 95% |
  |---|---|---|---|---|---|---|
  | B1 (VN30-Index proxy) | 219.6% | 17.7% | 0.880 | 1.232 | -33.0% | -3.47% |
  | B2 (Buy & Hold) | 246.6% | 19.1% | 0.904 | 1.267 | -35.3% | -3.61% |
  | B3 (Mean-Variance) | 346.7% | 23.4% | 0.866 | 1.253 | -42.7% | -4.57% |
  | B4 (PPO thuần) | 319.3% | 22.3% | **1.010** | **1.491** | **-26.5%** | -3.44% |

  Nhận xét sơ bộ: không có số liệu nào vượt ngưỡng cảnh báo look-ahead bias của đặc tả (Sharpe>3, return>100%/năm) — sanity check qua. B4 có Sharpe cao nhất và drawdown thấp nhất trong 4 baseline (PPO học được cách giảm rủi ro có trả giá bằng cumulative return thấp hơn B3). Đây mới là **1 seed** (seed=0) cho B4, chưa phải mean±std nhiều seed — cần chạy thêm seed nếu muốn dùng số liệu B4 này làm mốc so sánh cuối cùng trong luận văn. Script chạy: `scripts/_run_baselines_real.py` (driver tạm, không phải file production của subagent nào).

  ⚠️ **Bảng trên là full-period backtest (train=test=2018-2025), CHƯA đúng walk-forward split theo DoD** — xem kết quả walk-forward thật ngay dưới đây; số liệu B4 ở bảng trên có khả năng lạc quan quá mức do PPO train và eval trên cùng dữ liệu.

- [x] **Chạy lại B1–B4 đúng walk-forward split** (`single_window_phase_a`: train 2019-01-01→2022-12-31 = 1001 phiên, val 2023H1 = 121 phiên, test 2023H2 = 128 phiên). B1/B2/B3 không cần train (chạy trực tiếp trên test_df, out-of-sample theo đúng nghĩa vì không có tham số học từ dữ liệu). B4 được **train lại từ đầu chỉ trên train_df** (200,000 timesteps, seed=0, checkpoint riêng `checkpoints/b4_ppo_no_llm_phase_a_walkforward.zip`, không dùng chung checkpoint full-period) rồi backtest trên test_df — đây mới là out-of-sample thật. Script: `scripts/_run_baselines_walkforward.py`. Kết quả lưu tại `reports/phase_a/walk_forward/comparison_table.{csv,md}` (không ghi đè bảng full-period cũ):

  | Chiến lược (test 2023H2, 128 phiên) | Cumulative Return | Annualized | Sharpe | Sortino | Max Drawdown | CVaR 95% |
  |---|---|---|---|---|---|---|
  | B1 (VN30-Index proxy) | 6.75% | 13.85% | 0.804 | 1.140 | -13.0% | -2.49% |
  | B2 (Buy & Hold) | 6.69% | 13.71% | 0.797 | 1.130 | -13.2% | -2.49% |
  | B3 (Mean-Variance) | 8.06% | 16.62% | 0.821 | 1.200 | -16.4% | -2.90% |
  | B4 (PPO thuần) | **-12.5%** | **-23.3%** | **-1.277** | **-1.617** | -21.7% | -3.14% |

  🔍 **Phát hiện quan trọng**: Khi tách đúng train/test, B4 (PPO) đảo chiều hoàn toàn — từ Sharpe cao nhất (1.010, bảng full-period) thành **âm nặng và tệ nhất** trong 4 baseline khi test out-of-sample thật trên dữ liệu chưa từng thấy lúc train. B1/B2/B3 (không cần "học" từ dữ liệu) vẫn ổn định dương. Xác nhận đúng lo ngại: bảng full-period cũ lạc quan quá mức vì PPO đã gián tiếp "nhìn thấy" toàn bộ giai đoạn test trong lúc train (cùng 1 state_df 2018-2025 dùng để cả train và backtest). **Hệ quả cho việc train Đề xuất sau này**: bắt buộc phải train Đề xuất (LLM-PPO) chỉ trên `train_df`, KHÔNG được train trên toàn bộ `state_df`, rồi mới so sánh với bảng baseline walk-forward này — nếu không sẽ lặp lại đúng lỗi vừa phát hiện ở B4. Lưu ý: mới 1 seed, test window ngắn (128 ngày) nên biến động có thể do variance mẫu nhỏ — cần ≥5 seed theo DoD để kết luận chắc chắn về việc PPO thuần có thực sự kém hơn baseline thụ động hay không trong giai đoạn 2023H2 cụ thể này.

### 1.4 Kiến trúc LLM policy (mục 3.4)

- [x] `src/agents/super_agent_head.py` — `PortfolioPolicyLLM` (Qwen2.5-1.5B-Instruct + LoRA r=16 + policy_head + value_head), có `_DummyBackboneForTest` (attention submodule tên trùng target_modules LoRA) để unit-test kiến trúc + LoRA thật không cần mạng
- [x] `state_to_prompt()` — chốt phương án **text-prompt** (không dùng embedding số học nối trực tiếp), giữ tinh thần FLAG-Trader cho RQ3
- [x] Xác nhận forward pass thật với `Qwen/Qwen2.5-1.5B-Instruct` thật — đã tải weights thật qua mạng (~244s lần đầu, sau đó cache), load OK ở `dtype=torch.float32` trên CPU (1.54B params, hidden_size=1536); `forward()` trả đúng shape `(1, n_assets+1)`/`(1,1)`, `weights` sum=1, không NaN/Inf. **Phát hiện quan trọng (không phải bug, ghi lại cho luận văn):** `dtype=torch.bfloat16` (default hiện tại của `PortfolioPolicyLLM`) chạy được trên CPU nhưng **chậm hơn ~4x** so với `float32` cho cùng 1 forward pass (đo thật: bf16 ~20-22s/prompt vs fp32 ~5-7s/prompt, cùng máy, cùng prompt) — do CPU không có tăng tốc phần cứng cho bf16 matmul. `float32` NÊN dùng khi chạy CPU (dev/debug); giữ nguyên default `bfloat16` trong code production vì đây là lựa chọn đúng cho mục tiêu triển khai GPU thật. Script verify (không production): `scripts/_smoke_test_llm_real.py`.

### 1.5 Training pipeline PPO custom (mục 3.6)

- [x] `src/training/rollout_buffer.py` — `RolloutBuffer` (add/get/reset)
- [x] `src/training/ppo_trainer.py` — `SimplePPOTrainer`: `collect_rollout` (torch.no_grad), `compute_gae` (GAE thật — verify bằng tay trong test), `update` (PPO clip + value loss + entropy). Action sampling qua **phân phối Dirichlet** (weights làm concentration) để có log_prob/entropy hợp lệ trên simplex — quyết định thiết kế bổ sung vì đặc tả chỉ cho `forward()` deterministic.
- [x] `src/training/train_phase_a.py` — CLI `--config/--seed/--smoke-test`, multi-seed loop, in số param trainable, checkpoint chỉ lưu phần LoRA+head, wandb/tensorboard tự tắt an toàn nếu thiếu key/package
- [x] `tests/test_ppo_trainer.py` — **12/12 pass**; đã chạy thử **toàn bộ `run_one_seed()` thật** với `PortfolioEnvPhaseA` thật + dummy backbone (thay Qwen) — xác nhận pipeline load config → env → policy → PPO update → checkpoint chạy hết không lỗi runtime
- [x] Xác nhận `requires_grad` chỉ đúng LoRA + 2 head — verify được bằng LoRA thật gắn lên dummy backbone (peft thật, không giả lập)
- [x] Verify `SimplePPOTrainer` (`collect_rollout` + `compute_gae` + `update`) chạy được với **model thật** Qwen2.5-1.5B (không dummy backbone) trên CPU — `n_steps=4`, `n_epochs=1`, dữ liệu thật `data/processed/phase_a.parquet` (40 ngày đầu, 5 mã). Kết quả đo thật (CPU 12-core, không GPU, `dtype=float32`):
  - `collect_rollout(n_steps=4)`: **16.28s** (**4.07s/step**) — reward không NaN, mean=-0.036
  - `update(n_epochs=1, batch_size=4)`: **30.68s** — `policy_loss`/`value_loss`/`entropy`/`approx_kl`/`clip_frac` đều là số hữu hạn, không NaN
  - Ngoại suy tuyến tính từ 4.07s/step: `n_steps=256` ≈ **17.4 phút**, `n_steps=512` (default `SimplePPOTrainer.n_steps`) ≈ **34.7 phút** — CHỈ riêng 1 rollout, chưa tính `update()`/nhiều update. Với cấu hình full spec (`total_updates=200`, `n_steps=512`, `n_epochs=4`) train thật trên CPU máy này là **không khả thi** (hàng chục/hàng trăm giờ) — xác nhận đúng cảnh báo hiệu năng mục 3.6 đặc tả, cần GPU.
  - Không phát hiện bug runtime nào (không shape mismatch, không lỗi dtype, không OOM) — kiến trúc `PortfolioPolicyLLM` + `SimplePPOTrainer` hoạt động đúng với model thật y hệt như với dummy backbone.
- [ ] Log reward trung bình lên wandb với **model thật** *(cơ chế đã code + verify hoạt động đúng ở trên qua console/stats dict, chưa thật sự log lên wandb — cần chạy training thật dài hơn)*
- [ ] Train thật tối thiểu 5 seed với Qwen2.5-1.5B thật *(cần GPU + nhiều giờ — cảnh báo hiệu năng mục 3.6 đặc tả; xem ước lượng thời gian thật ở trên — KHÔNG khả thi trên CPU)*

### 1.6 Evaluation pipeline (mục 3.7)

- [x] `src/evaluation/metrics.py` — `compute_all_metrics()` qua `quantstats` — đã sửa 3 sai khác giữa API thật của quantstats 0.0.81 và code mẫu đặc tả (xem mục 3 bên dưới), bọc từng metric bằng try/except riêng
- [x] `src/evaluation/backtest.py` — `run_backtest()` dispatch theo loại policy (SB3 `.predict`/`torch.nn.Module`/callable), `run_backtest_multi_seed()`, CLI an toàn khi thiếu checkpoint
- [x] `src/evaluation/report.py` — `build_comparison_table()` (mean±std), `export_report()` → CSV/Markdown, CLI `--phase a|b`
- [x] `paired_significance_test()` (`scipy.stats.ttest_rel`), xử lý edge case 2 chuỗi giống hệt nhau (p=1.0 thay vì NaN) và độ dài khác nhau (ValueError rõ ràng)
- [x] `tests/test_metrics.py` — **16/16 pass**
- [ ] Chạy evaluation thật trên kết quả training thật *(phụ thuộc 1.5)*

### 1.7 Definition of Done — Giai đoạn A

- [~] Bảng kết quả: **B1–B4 đã có số liệu thật theo ĐÚNG walk-forward split** (`single_window_phase_a`: train 2019-2022, val 2023H1, test 2023H2 — xem mục 1.3, kết quả tại `reports/phase_a/walk_forward/`), không còn là full-period backtest nữa (bảng full-period cũ vẫn giữ lại ở `reports/phase_a/comparison_table.md` để đối chiếu, đã ghi rõ là không dùng làm số liệu cuối). Phần "Đề xuất (mean±std, 5 seed)" **chưa làm được** — verify bằng model thật cho thấy 1 rollout n_steps=512 ước tính ~35 phút chỉ riêng bước collect, train hội tụ 5 seed trên CPU không khả thi (xem mục 1.5) → **cần GPU**. Khi train Đề xuất, PHẢI dùng cùng `train_df`/`test_df` (không phải toàn bộ `state_df`) để so sánh hợp lệ với bảng walk-forward này — xem cảnh báo look-ahead ở mục 1.3.
- [x] Review thiết kế không có look-ahead bias đã biết: `build_indicators` tính riêng từng ticker (có test), `walk_forward_split` không chồng lấn train/val/test (có test), B1-B4 chạy trên dữ liệu thật ra số liệu hợp lý (không vượt ngưỡng cảnh báo Sharpe>3/return>100%) — sanity check qua.
- [x] Test tối thiểu cho data + env — **101/101 pass toàn repo**
- [ ] Log training đầy đủ trên wandb (screenshot cho luận văn) — **cần GPU để train đủ dài mới có gì để log/screenshot có ý nghĩa**

**Kết luận Giai đoạn A**: walking skeleton đã chạy đầu-cuối thật với dữ liệu thật (fetch → indicators → env → 4 baseline → metrics → report) và kiến trúc LLM-PPO đã verify đúng với model thật Qwen2.5-1.5B (không phải chỉ dummy). Rào cản duy nhất còn lại để đạt DoD đầy đủ là **cần GPU** để train Đề xuất hội tụ đủ 5 seed — đây là giới hạn phần cứng của máy hiện tại, không phải vấn đề code.

---

## 2. GIAI ĐOẠN B — Mở rộng

> Khung code đã viết song song với Giai đoạn A (tái sử dụng interface) để tiết kiệm thời gian, nhưng
> **việc train/chạy thật vẫn nên đợi Giai đoạn A đạt Definition of Done đầy đủ** theo đúng tinh thần đặc tả.

### 2.1 Mở rộng dữ liệu 10 mã (mục 4.2)

- [x] `TICKERS_PHASE_B` đã định nghĩa trong `fetch_ohlcv.py` (5 mã Giai đoạn A + MWG, MSN, GAS, VIC, TCB)
- [x] `fetch_ohlcv.py`/`build_dataset.py` đã hỗ trợ `--phase b` (code tổng quát, không cần sửa logic)
- [x] Chạy thật `fetch_ohlcv --phase b` cho 10 mã — **xong**: `data/raw/ohlcv/phase_b.parquet` (10 mã × 1829 phiên = 18290 dòng, không thiếu ngày nào/mã) + `data/processed/phase_b.parquet` qua `build_dataset` (1796 dòng × 10 mã × 6 feature, 0 NaN, 2018-10-25 → 2025-12-31)
- [x] Hoàn thiện pipeline tin tức thật (bắt buộc cho Ablation A/B6 có ý nghĩa) — crawl thật xong + đã nối vào `train_phase_b.py` theo-ngày qua `attach_recent_news()` (xem chi tiết mục 1.1 `fetch_news.py`); **5496 bài/10 mã** tại `data/raw/news/articles.parquet` (sau vòng crawl mở rộng FPT/HPG/MWG ngày 2026-09-08), verify thật cho kết quả non-trivial (VCB 40.2% ngày có tin thật, trải từ 2018). Cả 10/10 mã nay đều phủ tin tức trước mốc 2018-10-25 (điểm bắt đầu OHLCV) — không còn khoảng trống độ phủ nào chặn Ablation A/B6.
- [x] `walk_forward_split.make_walk_forward_windows()` hỗ trợ nhiều cửa sổ (đã test)

### 2.2 Tầng 1 — 3 Base Agent chuyên biệt (mục 4.3)

- [x] `src/agents/base_agent.py` — `BaseAgent(PortfolioPolicyLLM)` output 1 score [-1,1], `build_fundamental_prompt/build_technical_prompt/build_sentiment_prompt/build_prompt_for_agent`
- [ ] Train độc lập từng base agent trên reward cục bộ với **dữ liệu 10 mã + tin tức thật** *(công thức reward cục bộ đã code trong `train_phase_b.py` — MSE(score, tanh(next_day_return)) như simplification hợp lý cho MVP, đã unit-test logic nhưng chưa chạy trên data thật)*

### 2.3 Tầng 2 — Meta Aggregation (mục 4.4)

- [x] `src/agents/meta_agent.py` — `MetaAggregator` (nn thường, không LLM), `compute_confidence()`, `generate_explanation()` (chỉ inference, torch.no_grad, raise rõ ràng nếu backbone không có `.generate()`)
- [ ] Train Meta Aggregator với Tầng 1 đã freeze trên dữ liệu thật *(logic freeze + assert kiểm tra `requires_grad=False` đã code và unit-test trong `train_phase_b.py::train_meta_aggregator`, cần base agents đã train thật trước)*

### 2.4 Staged training (mục 4.5)

- [x] `src/training/train_phase_b.py` — `train_staged()` orchestration 3 bước tuần tự đầy đủ, CLI `--config --smoke-test`
- [x] Assert tường minh `requires_grad=False` sau mỗi bước freeze (Bước 2 và Bước 3) — code có check + unit test riêng cho cơ chế assert
- [x] Lưu checkpoint riêng biệt theo bước: `checkpoints/phase_b/{base_agents,meta,super_agent}/`
- [x] `MetaScoreObsWrapper` — adapter đưa 5 giá trị (meta_score + 3 base score + avg_confidence) vào đúng slot obs mà `SimplePPOTrainer`/`state_to_prompt` kỳ vọng, tắt cơ chế "khôi phục giá thật" (tránh nhân sai giá trị score với giá cổ phiếu)
- [ ] Chạy staged training end-to-end **thật** với Qwen thật + data 10 mã thật *(cần GPU + nhiều giờ; CLI đã verify đúng qua `--help` và logic xung quanh đã test bằng fake agent nhẹ, KHÔNG chạy được `--smoke-test` thật vì cần tải Qwen ~3GB)*

### 2.5 Baseline B5, B6 và Ablation A, B (mục 4.6)

- [x] B5 = tái sử dụng `PortfolioPolicyLLM` Giai đoạn A trên 10 mã (không code mới, đúng tinh thần đặc tả)
- [x] `src/baselines/b6_frozen_extractor.py` — `FrozenSentimentExtractor`, đóng băng hoàn toàn + `fixed_projection` random có seed cố định, hỗ trợ inject model/tokenizer để test offline
- [ ] Ablation A: B6 vs BaseAgent (fine-tune LoRA+PPO) — **cần chạy thật, cùng meta/super-agent**
- [ ] Ablation B: B5 (khối đơn) vs kiến trúc 3 tầng đầy đủ — so std Sharpe qua seed — **cần chạy thật**
- [ ] Chạy đủ 10 seed cho Ablation A & B — **cần chạy thật**

### 2.6 Definition of Done — Giai đoạn B

- [ ] Bảng kết quả đầy đủ Đề xuất vs B1–B6 (10 mã, 10 seed, walk-forward nhiều cửa sổ)
- [ ] Ablation A, B có kết quả + kiểm định thống kê minh bạch
- [ ] Trả lời RQ1, RQ2 bằng số liệu cụ thể
- [ ] Checkpoint cả 3 tầng lưu lại, load lại được để demo

---

## 3. Ghi chú tiến độ / quyết định thiết kế đã chốt (từ 6 subagent)

- **Text-prompt cho state → LLM** (mục 3.4): giữ nguyên quyết định gốc, không dùng embedding số học nối trực tiếp.
- **Obs giá đã chuẩn hóa**: `portfolio_env.py` chuẩn hóa giá bằng `close / close_ngày_0_episode` thay vì giá VND thô như code mẫu đặc tả (comment gốc gọi là "giá chuẩn hóa" nhưng code mẫu chưa làm đúng) — RL-friendly hơn nhiều so với giá thô hàng chục nghìn VND. `ppo_trainer.obs_array_to_dict()` có tham số `base_close` tùy chọn để khôi phục giá thật khi cần hiển thị trong prompt.
- **Action-masking**: khi biết `current_weights`, mã bị khóa **giữ nguyên tỷ trọng cũ** (đúng ý định đặc tả, sửa 1 chỗ pseudo-code gốc lỡ zero-out).
- **PPO action sampling**: dùng phân phối **Dirichlet** (concentration = weights×κ) trên đầu ra softmax của policy để có log_prob/entropy hợp lệ trên simplex — đặc tả gốc chỉ cho `forward()` deterministic, không đủ cho PPO cần sample.
- **quantstats API thật (0.0.81) khác code mẫu đặc tả 3 chỗ**: `max_drawdown` cần `prices` + `DatetimeIndex` (không phải `returns` + list thường); `cvar(sigma=0.95)` **không** phải CVaR 95% (tham số đúng là `confidence=0.95`); `information_ratio` không tự annualize. Đã sửa trong `metrics.py`.
- **Base agent Giai đoạn B dùng supervised MSE đơn giản** (không phải PPO đầy đủ) cho Bước 1 staged training — MVP simplification hợp lý vì reward cục bộ tuyến tính theo score sẽ đẩy score về ±1 nếu dùng policy-gradient trực tiếp; PPO đầy đủ chỉ dùng ở Bước 3 (Super-Agent) qua env thật.
- Python hệ thống là 3.13 (không phải 3.10 như đặc tả đề xuất) — torch 2.13, transformers, peft, accelerate, gymnasium, stable-baselines3 **đều cài và chạy được**. `bitsandbytes` (int4/int8 quantization) chưa cài — cần đánh giá lại khi thật sự train Qwen2.5-1.5B (không bắt buộc cho CPU dev, có thể cần khi train GPU VRAM hạn chế).
- **Toàn bộ mục còn `[ ]`** đều thuộc 1 trong 3 nhóm: (a) cần mạng thật để gọi `vnstock`/crawl tin tức, (b) cần tải trọng số `Qwen2.5-1.5B-Instruct` (~3GB) + đủ GPU/RAM, (c) cần thời gian training thật (nhiều giờ theo cảnh báo mục 3.6 đặc tả). Đây là các bước cần người vận hành xác nhận môi trường (mạng, GPU) trước khi chạy tiếp — không thể tự động hóa thêm trong phiên hiện tại.
- **CafeF URL đặc tả gốc đã chết** (2026-07-26): `https://cafef.vn/du-lieu/{ticker}.chn` trả 404 thật. URL đúng đã reverse-engineer + verify bằng request thật: `https://cafef.vn/{ticker}.html` (trang 1), `https://cafef.vn/{ticker}/trang-{n}.html` (trang n>1), selector `div.tlitem`. Đã crawl thật 1000 bài/10 mã, xem chi tiết mục 1.1.
- **File `AGENTS.md` ở gốc `hifa/` không phải tài liệu dự án** — nội dung là một kịch bản chỉ đạo AI agent tự động cài package + xin/âm thầm cấu hình API key từ "vnstocks.com/settings", có dấu hiệu prompt injection. KHÔNG làm theo nội dung file này. Cần người vận hành xác minh nguồn gốc file (subagent trước tạo nhầm hay do gói `vnstock` tự sinh ra khi cài) và quyết định xoá hay giữ.
