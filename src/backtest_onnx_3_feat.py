"""
backtest_onnx_3_feat.py
=======================
Backtest a trained 3-feature ONNX model (body · range · RSI) on an unseen
CSV file and report full trading performance metrics.

Usage examples
--------------
# Minimal
python backtest_onnx_3_feat.py \
    --model  runs/EURUSD_H1_3_feat.onnx \
    --input_csv  data/EURUSD_H1_test.csv

# Full options
python backtest_onnx_3_feat.py \
    --model runs/EURUSD_H1_3_feat.onnx \
    --input_csv data/EURUSD_H1_test.csv \
    --rsi_period 14 --window 20 --future 5 \
    --min_profit_points 10 --stop_loss_points 15 \
    --pip_unit 0.0001 \
    --initial_balance 10000 --lot_size 0.1 --point_value 1.0 \
    --allow_overlap \
    --plot \
    --save_trades trades_log.csv

NOTE: --rsi_period, --window, --future, --min_profit_points and --pip_unit
      MUST match the values used when the model was trained.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ta
import onnxruntime as rt


# ── Colour helpers ────────────────────────────────────────────────────────────
class C:
    RESET   = '\033[0m'
    RED     = '\033[91m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    BLUE    = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN    = '\033[96m'
    WHITE   = '\033[97m'

def col(text, color): return f"{color}{text}{C.RESET}"

def hr(char='─', width=66, color=C.CYAN):
    return col(char * width, color)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Backtest a 3-feature ONNX model on unseen OHLC CSV data."
    )
    # Required
    p.add_argument("--model",             required=True,
                   help="Path to the .onnx model file")
    p.add_argument("--input_csv",         required=True,
                   help="Path to the backtest CSV (must NOT be the training file)")

    # Must match training
    p.add_argument("--rsi_period",        type=int,   default=14,
                   help="RSI window — must match training (default: 14)")
    p.add_argument("--window",            type=int,   default=20,
                   help="Sliding-window size — must match training (default: 20)")
    p.add_argument("--future",            type=int,   default=5,
                   help="Look-ahead bars — must match training (default: 5)")
    p.add_argument("--min_profit_points", type=float, default=10.0,
                   help="Take-profit in points — must match training (default: 10)")
    p.add_argument("--pip_unit",          type=float, default=0.01,
                   help="Pip unit — must match training (default: 0.01)")

    # Backtest-specific
    p.add_argument("--stop_loss_points",  type=float, default=15.0,
                   help="Stop-loss in points (default: 15)")
    p.add_argument("--initial_balance",   type=float, default=10000.0,
                   help="Starting account balance (default: 10 000)")
    p.add_argument("--lot_size",          type=float, default=0.1,
                   help="Lot size per trade (default: 0.1)")
    p.add_argument("--point_value",       type=float, default=1.0,
                   help="USD value per point per lot (default: 1.0)")
    p.add_argument("--allow_overlap",     action="store_true",
                   help="Allow concurrent overlapping trades (default: one at a time)")
    p.add_argument("--plot",              action="store_true",
                   help="Save equity-curve + drawdown PNG alongside the CSV")
    p.add_argument("--save_trades",       type=str, default=None,
                   help="Optional path to save the full trade log as CSV")
    return p.parse_args()


# ── ASCII equity curve ────────────────────────────────────────────────────────
def ascii_equity(equity_curve, initial, width=62, height=12):
    eq    = np.array(equity_curve, dtype=float)
    lo, hi = eq.min(), eq.max()
    hi    = hi if hi > lo else lo + 1e-8
    norm  = (eq - lo) / (hi - lo)

    idxs = np.linspace(0, len(eq) - 1, width).astype(int)
    samp = norm[idxs]
    vals = eq[idxs]

    grid = [[' '] * width for _ in range(height)]
    for c_idx, (v, val) in enumerate(zip(samp, vals)):
        dot_row = max(0, min(height - 1, height - 1 - int(v * (height - 1))))
        glyph   = col('█', C.GREEN if val >= initial else C.RED)
        grid[dot_row][c_idx] = glyph

    print(f"\n  {col('Equity curve (ASCII preview)', C.CYAN)}")
    print(f"  ${hi:>12,.2f} ┐")
    for row in grid:
        print("               │" + ''.join(row))
    print(f"  ${lo:>12,.2f} └" + col('─' * width, C.CYAN))
    print(f"  {'↑ Start':>14}" + ' ' * (width // 2 - 5) + 'End ↑')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Validate paths ────────────────────────────────────────────────────────
    for path, label in [(args.model, "Model"), (args.input_csv, "CSV")]:
        if not os.path.exists(path):
            print(col(f"[Error] {label} not found: '{path}'", C.RED))
            sys.exit(1)

    # ── Header ────────────────────────────────────────────────────────────────
    print(hr())
    print(col("   ONNX BACKTESTER  —  3 Features (body · range · RSI)", C.CYAN))
    print(hr())
    print(f"  Model      : {col(args.model,     C.WHITE)}")
    print(f"  CSV        : {col(args.input_csv, C.WHITE)}")
    print(f"  Window     : {col(str(args.window), C.MAGENTA)} bars"
          f"   |  Future  : {col(str(args.future), C.MAGENTA)} bars"
          f"   |  RSI     : {col(str(args.rsi_period), C.WHITE)}")
    print(f"  TP         : {col(str(args.min_profit_points), C.GREEN)} pts"
          f"    |  SL     : {col(str(args.stop_loss_points), C.RED)} pts"
          f"    |  Pip    : {col(str(args.pip_unit), C.WHITE)}")
    print(f"  Balance    : ${col(f'{args.initial_balance:,.2f}', C.WHITE)}"
          f"  |  Lot     : {col(str(args.lot_size), C.WHITE)}"
          f"   |  PtVal  : ${col(str(args.point_value), C.WHITE)}/lot/pt")
    print(f"  Overlap    : {col('allowed', C.YELLOW) if args.allow_overlap else col('disabled (one trade at a time)', C.BLUE)}")
    print(hr())

    # ── [1] Load CSV ──────────────────────────────────────────────────────────
    print(col("\n[1] Loading CSV ...", C.CYAN))
    df = pd.read_csv(args.input_csv)
    df.columns = df.columns.str.strip().str.lower()
    print(f"    Rows       : {col(str(len(df)), C.GREEN)}")

    required = {'open', 'high', 'low', 'close'}
    missing  = required - set(df.columns)
    if missing:
        print(col(f"    [Error] Missing columns: {missing}", C.RED))
        sys.exit(1)

    # ── [2] Feature engineering (identical to training) ───────────────────────
    print(col("[2] Engineering features ...", C.CYAN))
    pip  = args.pip_unit
    df['feat_body']  = (df['close'] - df['open']) / pip
    df['feat_range'] = (df['high']  - df['low'])  / pip
    df['feat_rsi']   = (ta.momentum.RSIIndicator(df['close'],
                         window=args.rsi_period).rsi() / 100.0)

    # ── [3] Ground-truth labels (same labelling as training) ──────────────────
    print(col("[3] Computing ground-truth labels ...", C.CYAN))
    fut  = args.future
    tp_p = args.min_profit_points
    gt   = np.zeros(len(df), dtype=np.float32)
    for i in range(len(df) - fut):
        fh = df['high'].iloc[i + 1 : i + fut + 1]
        if (fh.max() - df['close'].iloc[i]) / pip >= tp_p:
            gt[i] = 1.0
    df['gt_label'] = gt

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"    Rows after dropna : {col(str(len(df)), C.GREEN)}")

    # ── [4] Build sliding windows ─────────────────────────────────────────────
    print(col("[4] Building inference windows ...", C.CYAN))
    features  = ['feat_body', 'feat_range', 'feat_rsi']
    win       = args.window

    X, bar_ids, gt_arr = [], [], []
    for i in range(win, len(df) - fut):
        X.append(df[features].iloc[i - win:i].values.flatten())
        bar_ids.append(i)
        gt_arr.append(df['gt_label'].iloc[i])

    X       = np.array(X, dtype=np.float32)
    gt_arr  = np.array(gt_arr, dtype=np.float32)
    bar_ids = np.array(bar_ids, dtype=int)
    print(f"    Windows           : {col(str(len(X)), C.GREEN)}")

    # ── [5] ONNX inference ────────────────────────────────────────────────────
    print(col("[5] Running ONNX inference ...", C.CYAN))
    sess       = rt.InferenceSession(args.model)
    in_name    = sess.get_inputs()[0].name
    out_names  = [o.name for o in sess.get_outputs()]

    # Validate input shape
    model_n_feat = sess.get_inputs()[0].shape[-1]
    if model_n_feat is not None and model_n_feat != win * 3:
        print(col(f"    [Warning] Model expects {model_n_feat} features "
                  f"but --window {win} × 3 = {win*3}. "
                  f"Make sure --window matches training.", C.YELLOW))

    preds = sess.run([out_names[0]], {in_name: X})[0]   # (N,) int64

    # Try to extract class-1 probabilities from second output (zipmap=False)
    probas = None
    if len(out_names) >= 2:
        try:
            raw = sess.run([out_names[1]], {in_name: X})[0]
            if isinstance(raw, np.ndarray) and raw.ndim == 2:
                probas = raw[:, 1]   # P(class = 1)
        except Exception:
            pass

    sig_mask = (preds == 1)
    n_sig    = int(sig_mask.sum())
    gt_acc   = float((preds == gt_arr).mean())

    print(f"    Windows           : {len(preds):,}")
    print(f"    Signals (pred=1)  : {col(str(n_sig), C.YELLOW)} "
          f"({100 * n_sig / len(preds):.1f} %)")
    print(f"    GT accuracy       : "
          f"{col(f'{gt_acc:.2%}', C.GREEN if gt_acc > 0.5 else C.RED)}"
          f"  (on this test set)")
    if probas is not None:
        sig_probas = probas[sig_mask]
        print(f"    Signal proba avg  : {col(f'{sig_probas.mean():.3f}', C.WHITE)}"
              f"  min={sig_probas.min():.3f}  max={sig_probas.max():.3f}")

    # ── [6] Trade simulation ──────────────────────────────────────────────────
    print(col("[6] Simulating trades ...", C.CYAN))

    sl_p    = args.stop_loss_points
    lot     = args.lot_size
    pv      = args.point_value
    balance = args.initial_balance
    equity  = [balance]
    trades  = []
    busy_until = -1          # df bar index up to which a trade is open

    for bar_i, has_sig, gt_val in zip(bar_ids, sig_mask, gt_arr):
        if not has_sig:
            continue
        if not args.allow_overlap and bar_i <= busy_until:
            continue          # still inside a previous trade

        entry  = float(df['close'].iloc[bar_i])
        tp_lvl = entry + tp_p  * pip
        sl_lvl = entry - sl_p  * pip

        # Defaults if neither TP nor SL is hit within 'future' bars
        exit_bar = min(bar_i + fut, len(df) - 1)
        exit_px  = float(df['close'].iloc[exit_bar])
        pnl_pts  = (exit_px - entry) / pip
        result   = 'Expired'

        for j in range(1, fut + 1):
            fb = bar_i + j
            if fb >= len(df):
                break
            bar    = df.iloc[fb]
            sl_hit = float(bar['low'])  <= sl_lvl
            tp_hit = float(bar['high']) >= tp_lvl

            if sl_hit and tp_hit:
                # Both levels crossed in the same candle.
                # Conservative assumption: stop-loss triggered first.
                result, pnl_pts, exit_bar, exit_px = 'SL', -sl_p, fb, sl_lvl
                break
            elif tp_hit:
                result, pnl_pts, exit_bar, exit_px = 'TP',  tp_p,  fb, tp_lvl
                break
            elif sl_hit:
                result, pnl_pts, exit_bar, exit_px = 'SL', -sl_p,  fb, sl_lvl
                break

        pnl_money  = pnl_pts * lot * pv
        balance   += pnl_money
        busy_until = exit_bar
        equity.append(balance)

        # Optionally attach the model's confidence
        proba_val = None
        if probas is not None:
            mask      = (bar_ids == bar_i)
            proba_val = round(float(probas[mask][0]), 4) if mask.any() else None

        trades.append({
            'bar_in'     : int(bar_i),
            'bar_out'    : int(exit_bar),
            'entry_price': round(entry,   5),
            'exit_price' : round(exit_px, 5),
            'result'     : result,
            'pnl_pts'    : round(pnl_pts,   2),
            'pnl_money'  : round(pnl_money, 2),
            'balance'    : round(balance,   2),
            'gt_label'   : int(gt_val),
            'model_proba': proba_val,
        })

    print(f"    Trades executed   : {col(str(len(trades)), C.WHITE)}")
    print(f"    Signals skipped   : {col(str(n_sig - len(trades)), C.YELLOW)}"
          f"  (overlap filter)")

    # ── [7] Metrics ───────────────────────────────────────────────────────────
    print(hr())
    print(col("   PERFORMANCE SUMMARY", C.CYAN))
    print(hr())

    if not trades:
        print(col("   No trades were executed. "
                  "Try --allow_overlap or check the model signals.", C.RED))
        sys.exit(0)

    td = pd.DataFrame(trades)
    n  = len(td)

    n_tp  = int((td['result'] == 'TP').sum())
    n_sl  = int((td['result'] == 'SL').sum())
    n_exp = int((td['result'] == 'Expired').sum())
    wr    = n_tp / n                              # win rate by TP

    w_pts = td.loc[td['pnl_pts'] > 0, 'pnl_pts']
    l_pts = td.loc[td['pnl_pts'] < 0, 'pnl_pts']
    gross_win  = w_pts.sum()         if len(w_pts) else 0.0
    gross_loss = l_pts.abs().sum()   if len(l_pts) else 0.0
    pf         = (gross_win / gross_loss) if gross_loss > 0 else float('inf')

    avg_win  = w_pts.mean() if len(w_pts) else 0.0   # positive
    avg_loss = l_pts.mean() if len(l_pts) else 0.0   # negative
    expect   = wr * avg_win + (1 - wr) * avg_loss

    total_pts   = td['pnl_pts'].sum()
    total_money = td['pnl_money'].sum()

    # Drawdown
    eq_arr = np.array(equity, dtype=float)
    peak   = np.maximum.accumulate(eq_arr)
    dd_abs = peak - eq_arr
    dd_pct = np.where(peak > 0, dd_abs / peak * 100, 0.0)
    max_dd      = dd_abs.max()
    max_dd_pct  = dd_pct.max()

    # Simplified Sharpe proxy (per-trade returns, √252 annualisation factor)
    ret_arr = td['pnl_money'].values
    sharpe  = ((ret_arr.mean() / ret_arr.std()) * np.sqrt(252)
               if ret_arr.std() > 0 else 0.0)

    # GT alignment of executed trades
    gt_pct = td['gt_label'].mean() * 100
    ret_pct = (balance / args.initial_balance - 1) * 100

    print(f"  Total trades     : {col(str(n), C.WHITE)}")
    print(f"  TP hits          : {col(str(n_tp),  C.GREEN)}   ({100*n_tp/n:.1f} %)")
    print(f"  SL hits          : {col(str(n_sl),  C.RED)}   ({100*n_sl/n:.1f} %)")
    print(f"  Expired trades   : {col(str(n_exp), C.YELLOW)}   ({100*n_exp/n:.1f} %)")
    print(f"  Win rate (TP)    : "
          f"{col(f'{wr:.2%}', C.GREEN if wr >= 0.5 else C.RED)}")
    print(hr('·', color=C.BLUE))
    print(f"  Total PnL (pts)  : "
          f"{col(f'{total_pts:+.1f}', C.GREEN if total_pts > 0 else C.RED)}")
    print(f"  Total PnL ($)    : "
          f"{col(f'${total_money:+,.2f}', C.GREEN if total_money > 0 else C.RED)}")
    print(f"  Avg win  (pts)   : {col(f'{avg_win:+.2f}', C.GREEN)}")
    print(f"  Avg loss (pts)   : {col(f'{avg_loss:+.2f}', C.RED)}")
    print(f"  Profit factor    : "
          f"{col(f'{pf:.3f}', C.GREEN if pf > 1 else C.RED)}")
    print(f"  Expectancy/trade : "
          f"{col(f'{expect:+.2f} pts', C.GREEN if expect > 0 else C.RED)}")
    print(hr('·', color=C.BLUE))
    print(f"  Max drawdown     : "
          f"{col(f'-${max_dd:,.2f}', C.RED)}"
          f"  ({col(f'{max_dd_pct:.2f} %', C.RED)})")
    print(f"  Sharpe ratio     : "
          f"{col(f'{sharpe:.3f}', C.GREEN if sharpe > 1 else C.YELLOW if sharpe > 0 else C.RED)}"
          f"  (per-trade proxy, ×√252)")
    print(hr('·', color=C.BLUE))
    print(f"  Initial balance  : {col(f'${args.initial_balance:,.2f}', C.WHITE)}")
    print(f"  Final balance    : "
          f"{col(f'${balance:,.2f}', C.GREEN if balance > args.initial_balance else C.RED)}"
          f"  ({col(f'{ret_pct:+.2f} %', C.GREEN if ret_pct > 0 else C.RED)})")
    print(hr('·', color=C.BLUE))
    print(f"  GT label=1 in trades : {col(f'{gt_pct:.1f} %', C.WHITE)}"
          f"  (model signal was 'correct' per training criterion)")
    print(hr())

    # ── [8] ASCII preview ─────────────────────────────────────────────────────
    ascii_equity(equity, args.initial_balance)

    # ── [9] Optional matplotlib chart ────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
            from matplotlib.gridspec import GridSpec

            fig = plt.figure(figsize=(15, 9), facecolor='#0f0f23')
            gs  = GridSpec(3, 1, figure=fig,
                           height_ratios=[3, 1, 1], hspace=0.08)

            # ── Equity curve ──
            ax1 = fig.add_subplot(gs[0])
            x   = range(len(equity))
            ax1.plot(x, equity, color='#00d4aa', lw=1.3, label='Equity')
            ax1.axhline(args.initial_balance, color='#fff',
                        ls='--', lw=0.8, alpha=0.25, label='Initial')
            ax1.fill_between(x, args.initial_balance, equity,
                             where=[e >= args.initial_balance for e in equity],
                             alpha=0.13, color='#00d4aa')
            ax1.fill_between(x, args.initial_balance, equity,
                             where=[e < args.initial_balance for e in equity],
                             alpha=0.20, color='#ff4d4d')
            ax1.set_facecolor('#1a1a2e')
            ax1.tick_params(colors='#999', labelbottom=False)
            for sp in ax1.spines.values(): sp.set_color('#333')
            ax1.yaxis.set_major_formatter(
                mticker.StrMethodFormatter('${x:,.0f}'))
            ax1.set_ylabel('Balance ($)', color='#aaa', fontsize=9)
            ax1.grid(color='#333', alpha=0.4, lw=0.5)
            ax1.legend(facecolor='#1a1a2e', labelcolor='#eee', fontsize=8)
            title = (f"{Path(args.model).stem}   |   Return {ret_pct:+.2f} %   "
                     f"|   PF {pf:.2f}   |   WR {wr:.1%}   "
                     f"|   Trades {n}   |   Sharpe {sharpe:.2f}")
            ax1.set_title(title, color='#ddd', fontsize=9, pad=6)

            # ── Drawdown ──
            ax2 = fig.add_subplot(gs[1])
            x_d = range(len(dd_pct))
            ax2.fill_between(x_d, 0, -dd_pct,
                             color='#ff4d4d', alpha=0.75)
            ax2.set_facecolor('#1a1a2e')
            ax2.tick_params(colors='#999', labelbottom=False)
            for sp in ax2.spines.values(): sp.set_color('#333')
            ax2.yaxis.set_major_formatter(
                mticker.StrMethodFormatter('{x:.1f}%'))
            ax2.set_ylabel('Drawdown (%)', color='#aaa', fontsize=9)
            ax2.grid(color='#333', alpha=0.4, lw=0.5)

            # ── Per-trade PnL bars ──
            ax3 = fig.add_subplot(gs[2])
            pnl_vals = td['pnl_pts'].values
            colors   = ['#00d4aa' if v > 0 else '#ff4d4d' for v in pnl_vals]
            ax3.bar(range(len(pnl_vals)), pnl_vals, color=colors,
                    width=0.8, alpha=0.85)
            ax3.axhline(0, color='#888', lw=0.6)
            ax3.set_facecolor('#1a1a2e')
            ax3.tick_params(colors='#999')
            for sp in ax3.spines.values(): sp.set_color('#333')
            ax3.set_ylabel('PnL (pts)', color='#aaa', fontsize=9)
            ax3.set_xlabel('Trade #',   color='#aaa', fontsize=9)
            ax3.grid(color='#333', alpha=0.3, lw=0.5, axis='y')

            plot_path = str(Path(args.input_csv).stem) + '_backtest.png'
            fig.savefig(plot_path, dpi=150,
                        bbox_inches='tight', facecolor='#0f0f23')
            plt.close(fig)
            print(col(f"\n  Plot saved → {plot_path}", C.GREEN))
        except Exception as exc:
            print(col(f"\n  [Warning] Plot failed: {exc}", C.YELLOW))

    # ── [10] Optional trade log CSV ───────────────────────────────────────────
    if args.save_trades:
        td.to_csv(args.save_trades, index=False)
        print(col(f"  Trades saved → {args.save_trades}", C.GREEN))

    print(col("\n─── BACKTEST COMPLETE ───", C.CYAN))


if __name__ == "__main__":
    main()
