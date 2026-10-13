"""src/baselines/c1_debate.py

Kiến trúc nền **C1 — Debate-style** (De-Cuong §6.4.1, §3.2 hướng AlphaAgents).

C1 là 1 trong 3 kiến trúc nền cần đánh giá thực nghiệm ở Chương 6 (cùng A1
FLAG-Trader-style và B1 HARLF-style). Đặc trưng của C1 theo đề cương:

    "nhiều LLM 'tranh luận' qua prompt tĩnh (không fine-tune), tổng hợp quyết
    định bằng heuristic/vote, không có hoặc có rất ít RL."

Vì KHÔNG fine-tune và KHÔNG học bằng RL, C1 là mốc đối chiếu cho:
    - RQ2 (so sánh 3 kiến trúc nền trên cùng điều kiện),
    - Ablation C ở Giai đoạn C (giá trị của việc THÊM Tầng 3 RL phía sau một
      cơ chế debate tĩnh — xem De-Cuong §10, B7).

THIẾT KẾ CÀI ĐẶT (tối giản, đúng tinh thần "debate tĩnh"):

1. Mỗi mã cổ phiếu được 3 "analyst agent" chuyên biệt chấm điểm độc lập
   (fundamental / technical / sentiment) — mỗi agent là 1 LLM ĐÓNG BĂNG sinh
   1 score ∈ [-1, 1] từ prompt chuyên môn (tái dùng prompt builders tiếng Anh
   ở `src/agents/base_agent.py`). Đây là "opening statements".

2. Một VÒNG TRANH LUẬN (debate round): mỗi agent được cho biết điểm trung
   bình của các agent khác (peer consensus) rồi điều chỉnh điểm của mình về
   phía đồng thuận một phần (hệ số ``revision_strength``) — mô phỏng
   cross-examination của AlphaAgents mà KHÔNG cần train. (Có thể đặt
   ``n_debate_rounds=0`` để tắt, thành thuần vote.)

3. TỔNG HỢP bằng heuristic (KHÔNG RL): điểm cuối mỗi mã = trung bình 3 agent
   sau debate. Chỉ giữ các mã có điểm > 0 (khuyến nghị mua), phân bổ tỷ trọng
   bằng ``softmax`` trên các điểm dương đó; nếu không mã nào dương -> giữ 100%
   tiền mặt. Áp trần tỷ trọng mỗi mã (``max_weight``) như môi trường Mỹ.

LƯU Ý vai trò LLM: để chạy được cả khi KHÔNG có model thật (unit test / CI),
`DebateAllocator` nhận một ``scorer`` — một callable
``(agent_type, ticker_data) -> float`` trả score ∈ [-1, 1]. Mặc định dùng
`LLMDebateScorer` (3 LLM đóng băng, giống B6). Test có thể tiêm scorer giả
tất định (deterministic) không cần mạng.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.agents.base_agent import AGENT_TYPES, build_prompt_for_agent

# Kiểu scorer: nhận (agent_type, ticker_data) -> score trong [-1, 1].
Scorer = Callable[[str, Dict[str, Any]], float]


# ---------------------------------------------------------------------------
# Scorer dùng LLM đóng băng thật (giống tinh thần B6 — không fine-tune)
# ---------------------------------------------------------------------------
class LLMDebateScorer:
    """Scorer cho C1 dùng LLM pretrained ĐÓNG BĂNG (không fine-tune, không RL).

    Dùng chung 1 backbone đóng băng cho cả 3 vai analyst (fundamental/
    technical/sentiment) qua 3 prompt chuyên môn khác nhau — đúng tinh thần
    "nhiều LLM tranh luận qua prompt tĩnh". Tái sử dụng `FrozenSentimentExtractor`
    (b6) làm bộ sinh score tĩnh để tránh lặp code tải/đóng băng backbone.
    """

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        projection_seed: int = 42,
    ) -> None:
        from src.baselines.b6_frozen_extractor import FrozenSentimentExtractor

        # 1 extractor đóng băng, dùng cho cả 3 vai (prompt khác nhau -> hidden
        # state khác nhau -> score khác nhau). Projection cố định (seed) để tái lập.
        self._extractor = FrozenSentimentExtractor(
            base_model_name=base_model_name,
            projection_seed=projection_seed,
            model=model,
            tokenizer=tokenizer,
        )

    def __call__(self, agent_type: str, ticker_data: Dict[str, Any]) -> float:
        prompt = build_prompt_for_agent(agent_type, ticker_data)
        return self._extractor.get_score(prompt)


# ---------------------------------------------------------------------------
# Bộ phân bổ Debate (heuristic, KHÔNG RL)
# ---------------------------------------------------------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    z = x - np.max(x)
    e = np.exp(z)
    s = e.sum()
    if s <= 1e-12:
        return np.full_like(x, 1.0 / x.size)
    return e / s


class DebateAllocator:
    """C1 — tổng hợp quyết định phân bổ bằng debate tĩnh + heuristic/vote.

    KHÔNG có tham số huấn luyện, KHÔNG RL. Phân bổ hoàn toàn quyết định bởi
    score của các analyst agent (LLM đóng băng) sau vài vòng tranh luận.
    """

    def __init__(
        self,
        tickers: Sequence[str],
        scorer: Optional[Scorer] = None,
        n_debate_rounds: int = 1,
        revision_strength: float = 0.5,
        max_weight: Optional[float] = 0.30,
        softmax_temp: float = 1.0,
        agent_types: Sequence[str] = AGENT_TYPES,
    ) -> None:
        """
        Args:
            tickers: rổ mã (thứ tự cố định).
            scorer: callable (agent_type, ticker_data) -> score ∈ [-1, 1].
                Mặc định None -> yêu cầu truyền scorer thật khi gọi
                ``allocate`` (vd. `LLMDebateScorer`). Test tiêm scorer giả.
            n_debate_rounds: số vòng tranh luận (0 = thuần vote, không debate).
            revision_strength: mỗi vòng, agent dịch điểm về phía đồng thuận
                của các agent khác bao nhiêu (0..1).
            max_weight: trần tỷ trọng mỗi mã (None = không áp trần).
            softmax_temp: nhiệt độ softmax khi biến điểm dương -> tỷ trọng.
            agent_types: danh sách vai analyst (mặc định 3 vai chuẩn).
        """
        self.tickers = list(tickers)
        self.scorer = scorer
        self.n_debate_rounds = int(n_debate_rounds)
        self.revision_strength = float(revision_strength)
        self.max_weight = None if max_weight is None else float(max_weight)
        self.softmax_temp = float(softmax_temp)
        self.agent_types = list(agent_types)

    def _debate_scores(self, per_agent: Dict[str, float]) -> float:
        """Chạy debate trên điểm của các agent cho 1 mã, trả điểm tổng hợp.

        per_agent: {agent_type: score ∈ [-1,1]}. Mỗi vòng, từng agent dịch về
        phía trung bình điểm các agent KHÁC một phần ``revision_strength``.
        Trả về trung bình các agent sau debate.
        """
        scores = dict(per_agent)
        types = list(scores.keys())
        for _ in range(self.n_debate_rounds):
            new_scores = {}
            for a in types:
                others = [scores[o] for o in types if o != a]
                peer = float(np.mean(others)) if others else scores[a]
                new_scores[a] = (
                    (1.0 - self.revision_strength) * scores[a]
                    + self.revision_strength * peer
                )
                # giữ trong [-1, 1]
                new_scores[a] = float(np.clip(new_scores[a], -1.0, 1.0))
            scores = new_scores
        return float(np.mean(list(scores.values()))) if scores else 0.0

    def _row_to_ticker_data(self, row, ticker: str) -> Dict[str, Any]:
        """Trích dữ liệu 1 mã từ 1 hàng state_df (MultiIndex (ticker, feature))."""
        def _get(feat, default=0.0):
            try:
                return float(row[(ticker, feat)])
            except Exception:  # noqa: BLE001
                return default

        data: Dict[str, Any] = {"ticker": ticker}
        for feat in ("pe", "pb", "eps_growth", "rsi_14", "macd", "macd_signal", "volatility_20d"):
            data[feat] = _get(feat)
        # recent_news là text (nếu có); fallback placeholder.
        try:
            news = row[(ticker, "recent_news")]
            data["recent_news"] = str(news) if news is not None else "No recent news."
        except Exception:  # noqa: BLE001
            data["recent_news"] = "No recent news."
        return data

    def allocate_one(self, row, scorer: Optional[Scorer] = None) -> np.ndarray:
        """Tính vector tỷ trọng (n_assets + 1,) cho MỘT ngày (1 hàng state_df)."""
        scorer = scorer or self.scorer
        if scorer is None:
            raise ValueError(
                "DebateAllocator cần `scorer` (callable (agent_type, ticker_data)"
                " -> float). Truyền qua __init__ hoặc tham số `scorer`."
            )

        final_scores = np.zeros(len(self.tickers), dtype=float)
        for i, tk in enumerate(self.tickers):
            ticker_data = self._row_to_ticker_data(row, tk)
            per_agent = {
                a: float(np.clip(scorer(a, ticker_data), -1.0, 1.0))
                for a in self.agent_types
            }
            final_scores[i] = self._debate_scores(per_agent)

        # Chỉ mua các mã điểm dương; phân bổ bằng softmax trên điểm dương.
        weights = np.zeros(len(self.tickers) + 1, dtype=float)  # +1 cash
        positive = final_scores > 0
        if not positive.any():
            weights[-1] = 1.0  # toàn tiền mặt
            return self._apply_cap(weights)

        pos_scores = final_scores[positive] / max(self.softmax_temp, 1e-6)
        alloc = _softmax(pos_scores)
        weights[np.where(positive)[0]] = alloc
        # phần còn lại (nếu softmax chuẩn hoá = 1 thì cash = 0) — ở đây cash=0.
        return self._apply_cap(weights)

    def _apply_cap(self, weights: np.ndarray) -> np.ndarray:
        """Áp trần tỷ trọng mỗi mã; phần vượt trần chảy vào CASH.

        Khác với `apply_position_cap` của env (phân bổ lại phần dư cho các mã
        chưa chạm trần), C1 cố ý dồn phần vượt trần vào tiền mặt: debate đã
        chọn tập mã mua (điểm dương) nên không "ép" vốn sang mã khác chỉ vì
        còn chỗ dưới trần — tránh bơm tỷ trọng cho mã debate không khuyến nghị.
        """
        w = np.asarray(weights, dtype=float).copy()
        s = w.sum()
        if s > 1e-12:
            w = w / s
        if self.max_weight is None:
            return w
        n_assets = w.shape[0] - 1
        for i in range(n_assets):
            if w[i] > self.max_weight:
                w[-1] += w[i] - self.max_weight  # dư vào cash
                w[i] = self.max_weight
        s = w.sum()
        return w / s if s > 1e-12 else w

    def predict(self, obs, state_df=None, t: Optional[int] = None):
        """Interface tương thích backtest: trả (weights, None).

        Hỗ trợ 2 cách gọi:
          - predict(row): ``obs`` là 1 hàng state_df (Series MultiIndex).
          - predict(obs_array, state_df=..., t=...): dùng state_df.iloc[t].
        """
        if state_df is not None and t is not None:
            row = state_df.iloc[t]
        else:
            row = obs
        return self.allocate_one(row), None


def run_c1_debate(
    state_df,
    tickers: Sequence[str],
    scorer: Optional[Scorer] = None,
    initial_cash: float = 1_000_000.0,
    transaction_cost: float = 0.0005,
    max_weight: Optional[float] = 0.30,
    n_debate_rounds: int = 1,
    revision_strength: float = 0.5,
) -> dict:
    """Chạy backtest C1 (Debate tĩnh) trên ``state_df``, trả kết quả + metric.

    Mỗi ngày: debate -> phân bổ -> cập nhật giá trị danh mục theo đúng công
    thức của môi trường Mỹ (net_return = gross_return - cost; cost =
    turnover * transaction_cost). KHÔNG train, KHÔNG RL.

    Returns:
        dict: portfolio_value_history, weights_history, turnover_history,
        dates, metrics.
    """
    from src.evaluation.metrics import compute_all_metrics

    tickers = list(tickers)
    allocator = DebateAllocator(
        tickers=tickers,
        scorer=scorer,
        n_debate_rounds=n_debate_rounds,
        revision_strength=revision_strength,
        max_weight=max_weight,
    )

    dates = state_df.index
    n_days = len(dates)
    weights_prev = np.zeros(len(tickers) + 1, dtype=float)
    weights_prev[-1] = 1.0  # bắt đầu 100% tiền mặt

    pv = [float(initial_cash)]
    weights_hist: List[np.ndarray] = []
    turnover_hist: List[float] = []

    for t in range(n_days - 1):
        row = state_df.iloc[t]
        next_row = state_df.iloc[t + 1]
        action = allocator.allocate_one(row)

        turnover = float(np.abs(action - weights_prev).sum())
        asset_returns = np.array(
            [
                (float(next_row[(tk, "close")]) - float(row[(tk, "close")]))
                / float(row[(tk, "close")])
                for tk in tickers
            ]
            + [0.0],
            dtype=float,
        )
        gross = float(np.dot(action, asset_returns))
        cost = turnover * transaction_cost
        net = gross - cost
        pv.append(pv[-1] * (1.0 + net))

        weights_hist.append(action)
        turnover_hist.append(turnover)
        weights_prev = action

    metrics = compute_all_metrics(pv, turnover_history=turnover_hist)

    return {
        "portfolio_value_history": np.asarray(pv),
        "weights_history": np.asarray(weights_hist) if weights_hist else np.zeros((0, len(tickers) + 1)),
        "turnover_history": np.asarray(turnover_hist),
        "dates": dates,
        "metrics": metrics,
    }
