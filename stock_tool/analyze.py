#!/usr/bin/env python3
"""
종목 확률 분석기 — 종목을 넣으면 '얼마나 오래 들고 있어야, 어떤 확률로, 얼마를 벌었나/벌 수 있나'를 계산한다.

⚠️ 이 도구는 '무조건 이기는' 도구가 아니다. 그런 도구는 존재할 수 없다.
   과거 가격 데이터로 확률분포를 추정할 뿐이며, 미래는 과거와 다를 수 있다.

사용 예:
    python analyze.py 000660                       # SK하이닉스 (6자리 코드는 .KS/.KQ 자동)
    python analyze.py NVDA --monthly 2000000 --lump 10000000
    python analyze.py 267260 --html report.html    # HTML 리포트 저장
    python analyze.py --csv prices.csv --name 테스트  # 오프라인: Date,Close 컬럼 CSV
    python analyze.py --demo                       # 가상 데이터로 동작 확인
"""
from __future__ import annotations

import argparse
import html
import math
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTH_DAYS = 21


# ─────────────────────────────── 데이터 로딩 ───────────────────────────────

def load_prices(ticker: str | None, csv: str | None, demo: bool) -> tuple[pd.Series, str, dict]:
    """(일별 수정종가 Series, 표시 이름, 부가정보 dict)를 돌려준다."""
    if demo:
        return _demo_prices(), "DEMO(가상종목)", {}
    if csv:
        df = pd.read_csv(csv)
        date_col = next(c for c in df.columns if c.lower() in ("date", "날짜", "일자"))
        close_col = next(c for c in df.columns if c.lower() in ("adj close", "close", "종가"))
        s = pd.Series(df[close_col].astype(float).values, index=pd.to_datetime(df[date_col]))
        return s.sort_index().dropna(), csv, {}

    import yfinance as yf  # 필요할 때만 import

    candidates = [ticker]
    if ticker.isdigit() and len(ticker) == 6:  # 한국 종목코드
        candidates = [f"{ticker}.KS", f"{ticker}.KQ"]
    for t in candidates:
        df = yf.download(t, period="max", auto_adjust=True, progress=False)
        if df is not None and not df.empty:
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            info = {}
            try:
                raw = yf.Ticker(t).info or {}
                info = {k: raw.get(k) for k in ("shortName", "longName", "trailingPE", "forwardPE",
                                                 "priceToBook", "currency", "sector")}
            except Exception:
                pass
            name = info.get("shortName") or info.get("longName") or t
            return close.dropna(), f"{name} ({t})", info
    sys.exit(f"'{ticker}' 데이터를 찾지 못했습니다. 야후파이낸스 티커 형식(예: 005930.KS, AAPL)을 확인하세요.")


def _demo_prices(years: int = 15, seed: int = 4) -> pd.Series:
    rng = np.random.default_rng(seed)
    n = years * TRADING_DAYS
    # 연 15% 드리프트, 연 30% 변동성 + 가끔 하락 레짐
    daily = rng.normal(0.15 / TRADING_DAYS, 0.30 / math.sqrt(TRADING_DAYS), n)
    crash_days = rng.choice(n, 3, replace=False)
    for d in crash_days:
        daily[d:d + 40] -= 0.004
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.Series(10000 * np.exp(np.cumsum(daily)), index=idx)


# ─────────────────────────────── 기본 지표 ───────────────────────────────

def basic_stats(p: pd.Series) -> dict:
    lr = np.log(p).diff().dropna()
    years = len(p) / TRADING_DAYS
    cagr = (p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1
    vol = lr.std() * math.sqrt(TRADING_DAYS)
    running_max = p.cummax()
    dd = p / running_max - 1
    ma200 = p.rolling(200).mean().iloc[-1] if len(p) >= 200 else float("nan")
    high_52w = p.iloc[-TRADING_DAYS:].max()
    delta = p.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain.iloc[-1] / loss.iloc[-1]) if loss.iloc[-1] > 0 else 100.0
    return {
        "start": p.index[0].date(), "end": p.index[-1].date(), "years": years,
        "last": p.iloc[-1], "cagr": cagr, "vol": vol, "mdd": dd.min(),
        "dd_now": dd.iloc[-1], "off_52w_high": p.iloc[-1] / high_52w - 1,
        "above_ma200": p.iloc[-1] > ma200 if not math.isnan(ma200) else None,
        "ma200_gap": p.iloc[-1] / ma200 - 1 if not math.isnan(ma200) else float("nan"),
        "rsi": rsi,
    }


# ────────────────────── 과거: 보유기간별 일시금 수익 확률 ──────────────────────

def rolling_lump(p: pd.Series, horizons_y: list[float]) -> list[dict]:
    out = []
    arr = p.values
    for h in horizons_y:
        d = int(round(h * TRADING_DAYS))
        if d >= len(arr) - 20:
            continue
        r = arr[d:] / arr[:-d] - 1
        out.append({"h": h, "n": len(r), "p_win": (r > 0).mean(), "median": np.median(r),
                    "p10": np.percentile(r, 10), "p90": np.percentile(r, 90),
                    "p_loss20": (r < -0.2).mean()})
    return out


# ────────────────────── 과거: 적립식(매월) 백테스트 ──────────────────────

def dca_backtest(p: pd.Series, months_list: list[int], monthly: float, lump: float) -> list[dict]:
    m = p.resample("MS").first().dropna()  # 매월 첫 거래일 가격
    last_daily = p
    out = []
    for months in months_list:
        if len(m) < months + 2:
            continue
        results = []
        for s in range(0, len(m) - months):
            buy_prices = m.iloc[s:s + months].values
            end_date = m.index[s + months]
            end_price = last_daily.loc[:end_date].iloc[-1]
            shares = lump / buy_prices[0] + (monthly / buy_prices).sum()
            invested = lump + monthly * months
            results.append((shares * end_price, invested))
        vals = np.array(results)
        profit = vals[:, 0] - vals[:, 1]
        out.append({"months": months, "n": len(vals), "invested": vals[0, 1],
                    "p_win": (profit > 0).mean(), "median": np.median(profit),
                    "p10": np.percentile(profit, 10), "p90": np.percentile(profit, 90)})
    return out


# ────────────────────── 미래: 몬테카를로 (블록 부트스트랩) ──────────────────────

def monte_carlo(p: pd.Series, months_list: list[int], monthly: float, lump: float,
                drift_keep: float, n_sims: int, seed: int = 42) -> list[dict]:
    """과거 일별 수익률을 21일 블록 단위로 재표본추출해 미래 경로를 만든다.
    drift_keep<1이면 과거 평균수익률(드리프트)을 그만큼만 인정 → 과거의 '대박'이 반복된다는 가정을 깎는다."""
    lr = np.log(p).diff().dropna().values
    if len(lr) > 15 * TRADING_DAYS:  # 너무 오래된 데이터 비중 과다 방지: 최근 15년
        lr = lr[-15 * TRADING_DAYS:]
    # '평균수익률'은 산술평균 기준으로 깎는다: 산술 = 로그평균 + 분산/2
    var = lr.var()
    mu_arith = lr.mean() + var / 2
    if mu_arith > 0:  # 좋은 과거만 깎는다 (나쁜 과거는 그대로 반영)
        lr = lr + (drift_keep * mu_arith - var / 2) - lr.mean()
    rng = np.random.default_rng(seed)
    max_m = max(months_list)
    n_blocks_avail = len(lr) - MONTH_DAYS
    starts = rng.integers(0, n_blocks_avail, size=(n_sims, max_m))
    # 각 시뮬레이션·각 달의 월 로그수익률 (21일 블록 합)
    csum = np.concatenate([[0.0], np.cumsum(lr)])
    monthly_lr = csum[starts + MONTH_DAYS] - csum[starts]
    price_path = np.exp(np.cumsum(monthly_lr, axis=1))  # 각 월말 가격(시작=1)
    buy_price = np.concatenate([np.ones((n_sims, 1)), price_path[:, :-1]], axis=1)  # 각 월초 가격

    out = []
    for months in months_list:
        shares = lump / 1.0 + (monthly / buy_price[:, :months]).sum(axis=1)
        final = shares * price_path[:, months - 1]
        invested = lump + monthly * months
        profit = final - invested
        # 경로 중 최대 평가손실(원금 대비)
        out.append({"months": months, "invested": invested, "p_win": (profit > 0).mean(),
                    "median": np.median(profit), "p10": np.percentile(profit, 10),
                    "p90": np.percentile(profit, 90), "p_loss20": (profit < -0.2 * invested).mean(),
                    "exp_return": np.median(final / invested) - 1})
    return out


# ────────────────────── 현재 상태 조건부: 지금 같은 때 샀다면? ──────────────────────

def conditional_now(p: pd.Series, fwd_days: int = TRADING_DAYS) -> dict | None:
    """현재 '고점 대비 낙폭'과 '200일선 위/아래'가 비슷했던 과거 시점들의 1년 뒤 성과."""
    if len(p) < 200 + fwd_days + 50:
        return None
    dd = p / p.cummax() - 1
    ma = p.rolling(200).mean()
    above = p > ma
    fwd = p.shift(-fwd_days) / p - 1
    now_dd, now_above = dd.iloc[-1], above.iloc[-1]
    band = 0.07
    mask = (dd.sub(now_dd).abs() <= band) & (above == now_above) & fwd.notna() & ma.notna()
    sample = fwd[mask]
    if len(sample) < 60:
        return None
    all_fwd = fwd.dropna()
    return {"n": len(sample), "p_win": (sample > 0).mean(), "median": sample.median(),
            "base_p_win": (all_fwd > 0).mean(), "base_median": all_fwd.median(),
            "dd_now": now_dd, "above": bool(now_above)}


# ─────────────────────────────── 종합 판단 ───────────────────────────────

@dataclass
class Verdict:
    hold_months: int | None
    entry: str
    grade: str
    notes: list[str] = field(default_factory=list)


def make_verdict(stats: dict, mc: list[dict], cond: dict | None, target: float) -> Verdict:
    hold = next((r["months"] for r in mc if r["p_win"] >= target), None)
    notes = []
    # 진입 방식
    if cond and cond["p_win"] < cond["base_p_win"] - 0.05:
        entry = "분할매수 (지금과 비슷한 상황에서 과거 1년 성과가 평균보다 나빴음)"
    elif stats["dd_now"] <= -0.25 and stats["above_ma200"] is False:
        entry = "분할매수 (하락추세 중 큰 낙폭 — 바닥 확인 전 한 번에 사지 말 것)"
    elif stats["dd_now"] <= -0.15:
        entry = "적극 분할매수 구간 (고점 대비 큰 조정, 과거 조건부 성과 양호)"
    elif stats["rsi"] >= 75 and stats["off_52w_high"] > -0.03:
        entry = "추격 자제 — 적립식만 (단기 과열: RSI 높고 52주 신고가 부근)"
    else:
        entry = "적립식 + 조정 시 추가매수"
    # 등급
    one_year = next((r for r in mc if r["months"] == 12), None)
    if hold is not None and hold <= 24 and one_year and one_year["p_loss20"] < 0.15:
        grade = "A (확률 우위 뚜렷)"
    elif hold is not None and hold <= 60:
        grade = "B (장기보유 전제 시 확률 우위)"
    else:
        grade = "C (장기보유로도 확률 우위 약함 — 비중 작게)"
    if stats["vol"] > 0.45:
        notes.append(f"연 변동성 {stats['vol']:.0%}: 매우 큼. 포트폴리오 비중 10% 이하 권장.")
    if stats["mdd"] < -0.6:
        notes.append(f"역사적 최대낙폭 {stats['mdd']:.0%}: 이 정도 하락을 견딜 수 있는 금액만 투자.")
    if stats["years"] < 7:
        notes.append(f"데이터가 {stats['years']:.1f}년뿐: 한 번의 강세장만 본 통계일 수 있음(신뢰도 낮음).")
    if stats["cagr"] > 0.3:
        notes.append(f"과거 연수익률 {stats['cagr']:.0%}는 예외적 — 같은 속도가 반복된다고 가정하면 안 됨.")
    return Verdict(hold, entry, grade, notes)


# ─────────────────────────────── 출력 ───────────────────────────────

def won(x: float) -> str:
    sign = "-" if x < 0 else "+"
    x = abs(x)
    if x >= 1e8:
        return f"{sign}{x / 1e8:.2f}억"
    return f"{sign}{x / 1e4:,.0f}만"


def plain_won(x: float) -> str:
    return "0원" if x == 0 else won(x).lstrip("+")


def horizon_label(months: int) -> str:
    return f"{months // 12}년" if months % 12 == 0 else f"{months}개월"


def build_report(name, stats, info, lump_hist, dca_hist, mc, cond, verdict, args) -> list[tuple[str, list[list[str]] | str]]:
    """섹션 목록: (제목, 표(행 리스트, 첫 행=헤더) 또는 문단)"""
    sec = []
    plan = f"일시금 {plain_won(args.lump)} + 매월 {plain_won(args.monthly)}"
    sec.append(("결론", "\n".join(filter(None, [
        f"종목: {name}",
        f"투자 방식: {plan}",
        f"등급: {verdict.grade}",
        (f"권장 최소 보유기간: {horizon_label(verdict.hold_months)} "
         f"(이익 확률 {args.target:.0%} 이상이 되는 가장 짧은 기간)") if verdict.hold_months
        else f"권장 보유기간: 시뮬레이션한 최장 기간 안에 이익 확률 {args.target:.0%}에 도달하지 못함",
        f"진입 방법: {verdict.entry}",
        *_headline(mc, lump_hist, verdict),
        *(f"주의: {n}" for n in verdict.notes),
    ]))))

    rows = [["보유기간", "투입원금", "이익확률", "예상손익(중앙값)", "나쁜경우(하위10%)", "좋은경우(상위10%)", "-20% 이상 손실확률"]]
    for r in mc:
        rows.append([horizon_label(r["months"]), plain_won(r["invested"]), f"{r['p_win']:.0%}",
                     won(r["median"]), won(r["p10"]), won(r["p90"]), f"{r['p_loss20']:.0%}"])
    sec.append((f"미래 시뮬레이션 — {plan} (몬테카를로 {args.sims:,}회, 과거 평균수익률의 {args.drift_keep:.0%}만 인정)", rows))

    rows = [["보유기간", "이익확률", "중앙 수익률", "하위10%", "상위10%", "-20% 이상 손실확률", "표본 수"]]
    for r in lump_hist:
        rows.append([f"{r['h']:g}년", f"{r['p_win']:.0%}", f"{r['median']:+.0%}", f"{r['p10']:+.0%}",
                     f"{r['p90']:+.0%}", f"{r['p_loss20']:.0%}", f"{r['n']:,}"])
    sec.append(("과거 실적 — 아무 날에나 일시금으로 샀다면", rows))

    if dca_hist:
        rows = [["적립기간", "투입원금", "이익확률", "손익(중앙값)", "하위10%", "상위10%", "표본 수"]]
        for r in dca_hist:
            rows.append([horizon_label(r["months"]), plain_won(r["invested"]), f"{r['p_win']:.0%}",
                         won(r["median"]), won(r["p10"]), won(r["p90"]), f"{r['n']}"])
        sec.append((f"과거 실적 — {plan} 방식으로 과거 아무 달에 시작했다면", rows))

    rows = [["항목", "값"],
            ["데이터 기간", f"{stats['start']} ~ {stats['end']} ({stats['years']:.1f}년)"],
            ["현재가", f"{stats['last']:,.2f}"],
            ["연평균 수익률(CAGR)", f"{stats['cagr']:+.1%}"],
            ["연 변동성", f"{stats['vol']:.1%}"],
            ["역사적 최대낙폭", f"{stats['mdd']:.1%}"],
            ["현재 고점 대비", f"{stats['dd_now']:+.1%}"],
            ["52주 고점 대비", f"{stats['off_52w_high']:+.1%}"],
            ["200일선 대비", f"{stats['ma200_gap']:+.1%}"],
            ["RSI(14)", f"{stats['rsi']:.0f}"]]
    for k, label in (("trailingPE", "PER(후행)"), ("forwardPE", "PER(선행)"), ("priceToBook", "PBR")):
        if info.get(k):
            rows.append([label, f"{info[k]:.1f}"])
    sec.append(("현재 상태", rows))

    if cond:
        sec.append(("지금과 비슷한 상황에서 샀던 과거 사례 (1년 뒤)", "\n".join([
            f"조건: 고점 대비 {cond['dd_now']:+.0%}(±7%p), 200일선 {'위' if cond['above'] else '아래'} — 과거 {cond['n']:,}일",
            f"이 조건에서 1년 뒤 이익확률 {cond['p_win']:.0%} (전체 평균 {cond['base_p_win']:.0%})",
            f"이 조건에서 1년 뒤 중앙 수익률 {cond['median']:+.0%} (전체 평균 {cond['base_median']:+.0%})",
        ])))

    sec.append(("반드시 읽을 것", "\n".join([
        "1) '무조건 이기는' 분석은 존재하지 않습니다. 위 확률은 과거 가격 움직임이 미래에도 비슷하다는 가정 위의 추정치입니다.",
        "2) 생존편향: 지금 검색되는 종목은 살아남은 종목입니다. 상장폐지·장기부진 종목은 통계에 없습니다.",
        "3) 실적·산업 사이클·금리·전쟁 같은 펀더멘털 변화는 가격 통계에 미리 반영되지 않습니다.",
        "4) 환율(해외주식), 세금, 수수료는 계산에 없습니다.",
        "5) 이 도구는 '얼마나 오래, 어떤 방식으로 들고 가야 확률이 내 편이 되는지'를 보는 용도입니다. 매수 여부는 본인 판단입니다.",
    ])))
    return sec


def _headline(mc, lump_hist, verdict) -> list[str]:
    """결론에 넣을 '몇 년 들고 가면 몇 % 확률로 얼마' 문장."""
    pick = next((r for r in mc if r["months"] == verdict.hold_months), None) or max(mc, key=lambda r: r["p_win"])
    lines = [f"→ {horizon_label(pick['months'])} 동안 넣고 보유 시: 이익확률 {pick['p_win']:.0%}, "
             f"예상손익 {won(pick['median'])} (나쁘면 {won(pick['p10'])}, 좋으면 {won(pick['p90'])}), "
             f"원금 {plain_won(pick['invested'])}"]
    past5 = next((r for r in lump_hist if r["h"] == 5), None)
    fut5 = next((r for r in mc if r["months"] == 60), None)
    if past5 and fut5 and past5["p_win"] - fut5["p_win"] >= 0.15:
        lines.append(f"※ 과거엔 5년 보유 시 {past5['p_win']:.0%} 이겼지만 그건 '한 번 일어난 역사'입니다. "
                     f"같은 변동성으로 미래를 보수적으로 돌리면 {fut5['p_win']:.0%}입니다. 과거 승률을 그대로 믿지 마세요.")
    return lines


def print_report(sections) -> None:
    for title, body in sections:
        print("\n" + "═" * 78)
        print(f"■ {title}")
        print("─" * 78)
        if isinstance(body, str):
            print(body)
        else:
            widths = [max(_w(row[i]) for row in body) for i in range(len(body[0]))]
            for j, row in enumerate(body):
                print("  ".join(_pad(c, widths[i]) for i, c in enumerate(row)))
                if j == 0:
                    print("  ".join("-" * w for w in widths))
    print()


def _w(s: str) -> int:  # 한글 폭 2칸
    return sum(2 if ord(ch) > 0x1100 else 1 for ch in s)


def _pad(s: str, w: int) -> str:
    return s + " " * (w - _w(s))


def write_html(sections, name: str, path: str) -> None:
    parts = []
    for title, body in sections:
        parts.append(f"<section><h2>{html.escape(title)}</h2>")
        if isinstance(body, str):
            parts.append("".join(f"<p>{html.escape(line)}</p>" for line in body.split("\n")))
        else:
            head = "".join(f"<th>{html.escape(c)}</th>" for c in body[0])
            rows = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in body[1:])
            parts.append(f"<div class='t'><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>")
        parts.append("</section>")
    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>종목 확률 분석</title>
<style>
:root{{--bg:#fff;--fg:#1a1a1a;--mute:#666;--line:#e3e3e3;--head:#f5f5f5}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141414;--fg:#eee;--mute:#aaa;--line:#333;--head:#1f1f1f}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,sans-serif;max-width:960px;margin:0 auto;padding:16px}}
h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px;border-bottom:1px solid var(--line);padding-bottom:6px}}
p{{margin:4px 0}} .t{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right;white-space:nowrap}}
th{{background:var(--head)}} th:first-child,td:first-child{{text-align:left}}
</style></head><body><h1>{html.escape(name)} — 확률 분석</h1>{''.join(parts)}</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


# ─────────────────────────────── main ───────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="종목 확률 분석기")
    ap.add_argument("ticker", nargs="?", help="예: 000660, 005930.KS, NVDA")
    ap.add_argument("--monthly", type=float, default=2_000_000, help="매월 적립액(원), 기본 200만")
    ap.add_argument("--lump", type=float, default=0, help="처음 한 번에 넣는 금액(원), 기본 0")
    ap.add_argument("--target", type=float, default=0.8, help="권장 보유기간 기준 이익확률, 기본 0.8")
    ap.add_argument("--drift-keep", type=float, default=0.5,
                    help="미래 시뮬레이션에서 과거 평균수익률을 몇 %% 인정할지(0~1), 기본 0.5(보수적)")
    ap.add_argument("--sims", type=int, default=10_000, help="몬테카를로 횟수")
    ap.add_argument("--csv", help="오프라인 가격 CSV (Date, Close 컬럼)")
    ap.add_argument("--name", help="CSV 사용 시 표시 이름")
    ap.add_argument("--demo", action="store_true", help="가상 데이터로 실행")
    ap.add_argument("--html", help="HTML 리포트 저장 경로")
    args = ap.parse_args()
    if not (args.ticker or args.csv or args.demo):
        ap.error("종목코드를 넣거나 --csv / --demo 를 지정하세요.")

    prices, name, info = load_prices(args.ticker, args.csv, args.demo)
    if args.name:
        name = args.name
    if len(prices) < TRADING_DAYS * 2:
        sys.exit("데이터가 2년 미만이라 확률 분석이 무의미합니다.")

    stats = basic_stats(prices)
    lump_hist = rolling_lump(prices, [0.25, 0.5, 1, 2, 3, 5, 10])
    months_list = [6, 12, 24, 36, 60, 120]
    dca_hist = dca_backtest(prices, months_list, args.monthly, args.lump)
    mc = monte_carlo(prices, months_list, args.monthly, args.lump, args.drift_keep, args.sims)
    cond = conditional_now(prices)
    verdict = make_verdict(stats, mc, cond, args.target)

    sections = build_report(name, stats, info, lump_hist, dca_hist, mc, cond, verdict, args)
    print_report(sections)
    if args.html:
        write_html(sections, name, args.html)
        print(f"HTML 리포트 저장: {args.html}")


if __name__ == "__main__":
    main()
