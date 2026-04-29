"""
Correlated executor: reads log/inference_log.csv to detect signal correlations across symbols,
calculates 7-factor scores, and executes on the highest-scoring symbol when signals align.

Dry-run mode prevents trades and logs "WOULD_EXECUTE" instead.

Usage:
    python execute_correlated_onnx_on_mt5.py \
        --symbols "NAS100,SP500,DOW30" \
        --lot 1.0 --interval 5 \
        --sl_mult 1.5 --tp_mult 2.0 --dry_run
"""

import argparse
import csv
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple

import MetaTrader5 as mt5
import pandas as pd

from utils.colors import Colors, colorize as c
from utils.telegram import notify

# ---------------------------------------------------------------------------
# Paths & field definitions
# ---------------------------------------------------------------------------
_INFERENCE_LOG_FILE = os.path.join('log', 'inference_log.csv')     # written by upstream inference process
_LOG_FILE           = os.path.join('log', 'correlated_trading_log.csv')
_CORR_INF_LOG_FILE  = os.path.join('log', 'correlated_inference_log.csv')
_CORR_STATE_LOG_FILE = os.path.join('log', 'correlation_state_log.csv')

_LOG_FIELDS = [
    'timestamp', 'event', 'selected_symbol', 'direction',
    'price', 'sl', 'tp_target', 'atr', 'score', 'alignment', 'reason',
]

_CORR_INF_FIELDS = [
    'timestamp', 'symbol', 'p_buy', 'p_sell', 'p_hold',
    'signal_decision', 'score', 'alignment', 'consecutive_count',
    'atr_value', 'volume', 'close_price',
]



# Reference ATR (index points) used to normalise the atr_norm factor.
# Adjust if your instruments have very different typical ATR ranges.
_ATR_REFERENCE = 100.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class LatestInference:
    """Snapshot of the most recent inference row for one symbol."""
    symbol:          str
    timestamp:       pd.Timestamp      # kept as pandas Timestamp to preserve tz
    p_buy:           float
    p_sell:          float
    p_hold:          float
    signal_decision: str
    atr_value:       float
    volume:          float
    close_price:     float


@dataclass
class SymbolState:
    """Mutable per-symbol tracking state updated every loop iteration."""
    consecutive_count: int = 0
    last_signal:       str = ''        # 'BUY', 'SELL', or ''


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)


def _log(event: str, selected_symbol: str = '', **kwargs) -> None:
    """Append one row to the correlated trading log."""
    _ensure_dir(_LOG_FILE)
    exists = os.path.exists(_LOG_FILE)
    with open(_LOG_FILE, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
        if not exists:
            w.writeheader()
        row = {k: '' for k in _LOG_FIELDS}
        row.update({
            'timestamp':       datetime.now().isoformat(),
            'event':           event,
            'selected_symbol': selected_symbol,
        })
        row.update(kwargs)
        w.writerow(row)


def _log_inference(
    inferences: Dict[str, LatestInference],
    scores:     Dict[str, float],
    states:     Dict[str, SymbolState],
    alignment:  str,
) -> None:
    """Append one row per symbol to the correlated inference log."""
    _ensure_dir(_CORR_INF_LOG_FILE)
    exists = os.path.exists(_CORR_INF_LOG_FILE)
    now_str = datetime.now().isoformat()
    with open(_CORR_INF_LOG_FILE, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_CORR_INF_FIELDS)
        if not exists:
            w.writeheader()
        for symbol, inf in inferences.items():
            w.writerow({
                'timestamp':        now_str,
                'symbol':           symbol,
                'p_buy':            inf.p_buy,
                'p_sell':           inf.p_sell,
                'p_hold':           inf.p_hold,
                'signal_decision':  inf.signal_decision,
                'score':            round(scores.get(symbol, 0.0), 5),
                'alignment':        alignment,
                'consecutive_count': states[symbol].consecutive_count,
                'atr_value':        inf.atr_value,
                'volume':           inf.volume,
                'close_price':      inf.close_price,
            })


def _log_correlation_state(
    inferences: Dict[str, LatestInference],
    scores: Dict[str, float],
    alignment: str,
    best_symbol: Optional[str],
    best_score: float,
    action: str,
    symbols: list,
) -> None:
    """Log full correlation state snapshot for analysis."""
    _ensure_dir(_CORR_STATE_LOG_FILE)
    exists = os.path.exists(_CORR_STATE_LOG_FILE)
    
    # Build dynamic fields based on symbols
    fields = ['timestamp', 'alignment', 'best_symbol', 'best_score', 'action']
    for sym in symbols:
        fields.extend([f'{sym.lower()}_signal', f'{sym.lower()}_score', 
                      f'{sym.lower()}_p_buy', f'{sym.lower()}_p_sell', f'{sym.lower()}_atr'])
    
    row = {k: '' for k in fields}
    row['timestamp'] = datetime.now().isoformat()
    row['alignment'] = alignment
    row['best_symbol'] = best_symbol or ''
    row['best_score'] = round(best_score, 5) if best_symbol else ''
    row['action'] = action
    
    for symbol in symbols:
        if symbol in inferences:
            inf = inferences[symbol]
            row[f'{symbol.lower()}_signal'] = inf.signal_decision
            row[f'{symbol.lower()}_score'] = round(scores.get(symbol, 0.0), 5)
            row[f'{symbol.lower()}_p_buy'] = round(inf.p_buy, 5)
            row[f'{symbol.lower()}_p_sell'] = round(inf.p_sell, 5)
            row[f'{symbol.lower()}_atr'] = round(inf.atr_value, 2)
    
    with open(_CORR_STATE_LOG_FILE, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Inference log reader
# ---------------------------------------------------------------------------
def read_latest_inferences(symbols: list, max_age_sec: int = 30) -> Dict[str, LatestInference]:
    """
    Read the most-recent row per symbol from inference_log.csv.

    Timezone handling: all timestamps are compared within the same timezone
    space as the log file itself. If the log uses tz-aware timestamps, 'now'
    is also made tz-aware using that same tzinfo; if the log is naive, both
    sides are naive. This avoids the TypeError that arises from mixing the two.
    """
    if not os.path.exists(_INFERENCE_LOG_FILE):
        return {}

    try:
        df = pd.read_csv(_INFERENCE_LOG_FILE)
        if df.empty:
            return {}

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Derive 'now' in the same tz space as the log file.
        sample_tz = df['timestamp'].iloc[-1].tzinfo
        now = pd.Timestamp.now(tz=sample_tz) if sample_tz is not None else pd.Timestamp.now()

        latest: Dict[str, LatestInference] = {}
        for symbol in symbols:
            sym_df = df[df['symbol'] == symbol].sort_values('timestamp')
            if sym_df.empty:
                continue

            row = sym_df.iloc[-1]
            ts  = row['timestamp']

            if (now - ts).total_seconds() > max_age_sec:
                continue

            latest[symbol] = LatestInference(
                symbol          = symbol,
                timestamp       = ts,
                p_buy           = float(row['p_buy']),
                p_sell          = float(row['p_sell']),
                p_hold          = float(row['p_hold']),
                signal_decision = str(row['signal_decision']),
                atr_value       = float(row['atr_value']),
                volume          = float(row['volume']),
                close_price     = float(row['close_price']),
            )

        return latest

    except Exception as e:
        print(f"{c(Colors.RED, f'[ERROR] reading inference log: {e}')}")
        return {}


# ---------------------------------------------------------------------------
# 7-factor scoring
# ---------------------------------------------------------------------------
def calculate_score(
    inf:               LatestInference,
    consecutive_count: int   = 1,
    atr_reference:     float = _ATR_REFERENCE,
) -> float:
    """
    7-factor score (weights sum to 1.0):

    Factor                  Weight  Description
    ─────────────────────── ──────  ───────────────────────────────────────────
    1. signal_strength       0.25   max(p_buy,p_sell) − min(p_buy,p_sell)
                                    Spread between the dominant and recessive
                                    class probabilities.
    2. probability           0.25   max(p_buy, p_sell)
                                    Raw confidence of the dominant class.
    3. filter_passed_bonus   0.20   1.0 if BUY/SELL else 0.0
                                    Rewards signals that cleared all filters.
    4. volume_norm           0.10   min(volume / 1_000_000, 1.0)
                                    Higher volume = more reliable signal.
    5. consec_norm           0.10   min(consecutive_count / 5, 1.0)
                                    Rewards sustained, stable signal direction.
    6. hold_penalty          0.05   1.0 − p_hold
                                    Lower model indecision = better signal.
    7. atr_norm              0.05   min(atr_value / atr_reference, 1.0)
                                    Higher ATR = more tradeable volatility.
    """
    signal_strength      = max(inf.p_buy, inf.p_sell) - min(inf.p_buy, inf.p_sell)
    probability          = max(inf.p_buy, inf.p_sell)
    filter_passed_bonus  = 1.0 if inf.signal_decision in ('BUY', 'SELL') else 0.0
    volume_norm          = min(inf.volume / 1_000_000.0, 1.0)
    consec_norm          = min(consecutive_count / 5.0, 1.0)
    hold_penalty         = 1.0 - inf.p_hold
    atr_norm             = min(inf.atr_value / atr_reference, 1.0)

    return (
        signal_strength     * 0.25 +
        probability         * 0.25 +
        filter_passed_bonus * 0.20 +
        volume_norm         * 0.10 +
        consec_norm         * 0.10 +
        hold_penalty        * 0.05 +
        atr_norm            * 0.05
    )


# ---------------------------------------------------------------------------
# Per-symbol state tracking
# ---------------------------------------------------------------------------
def update_states(
    states:     Dict[str, SymbolState],
    inferences: Dict[str, LatestInference],
) -> None:
    """
    Update consecutive_count and last_signal for every symbol.

    Rules:
    - BUY/SELL that matches last_signal → increment consecutive_count.
    - BUY/SELL that differs from last_signal → reset to 1, record new signal.
    - HOLD or any non-actionable decision → reset streak to 0.
    """
    for symbol, inf in inferences.items():
        state = states[symbol]
        sig   = inf.signal_decision

        if sig in ('BUY', 'SELL'):
            if sig == state.last_signal:
                state.consecutive_count += 1
            else:
                state.consecutive_count = 1
                state.last_signal       = sig
        else:
            # HOLD or unexpected value — reset streak
            state.consecutive_count = 0
            state.last_signal       = ''


# ---------------------------------------------------------------------------
# Alignment and symbol selection
# ---------------------------------------------------------------------------
def check_alignment(inferences: Dict[str, LatestInference]) -> Tuple[bool, str]:
    """Return (is_aligned, alignment_label)."""
    signals = [
        inf.signal_decision
        for inf in inferences.values()
        if inf.signal_decision in ('BUY', 'SELL')
    ]
    if not signals:
        return False, 'NO_SIGNAL'

    buy_count  = sum(1 for s in signals if s == 'BUY')
    sell_count = sum(1 for s in signals if s == 'SELL')

    if buy_count == len(signals):
        return True, 'ALL_BUY'
    if sell_count == len(signals):
        return True, 'ALL_SELL'
    return False, f'MIXED({buy_count}B_{sell_count}S)'


def select_best_symbol(
    inferences: Dict[str, LatestInference],
    states:     Dict[str, SymbolState],
) -> Optional[Tuple[str, float, str]]:
    """
    Return (best_symbol, score, alignment_label) when signals are aligned,
    or None when they are not.
    """
    aligned, alignment = check_alignment(inferences)
    if not aligned:
        return None

    scores      = {sym: calculate_score(inf, states[sym].consecutive_count)
                   for sym, inf in inferences.items()}
    best_symbol = max(scores, key=scores.get)
    return best_symbol, scores[best_symbol], alignment


# ---------------------------------------------------------------------------
# MT5 helpers
# ---------------------------------------------------------------------------
def get_open_position(symbol: str):
    positions = mt5.positions_get(symbol=symbol)
    return positions[0] if positions else None


def open_position(
    symbol:    str,
    direction: str,
    lot:       float,
    sl_mult:   float,
    tp_mult:   float,
    atr:       float,
    magic:     int,
    deviation: int,
    dry_run:   bool,
) -> None:
    """Open a market position, or log/notify if dry_run is True."""
    sl_offset = atr * sl_mult
    tp_offset = atr * tp_mult

    if dry_run:
        _log(
            'WOULD_EXECUTE', symbol,
            direction  = direction,
            atr        = round(atr, 5),
            sl         = f'±{round(sl_offset, 5)}',
            tp_target  = f'±{round(tp_offset, 5)}',
            reason     = 'DRY_RUN',
        )
        notify(f"[DRY] {symbol} {direction}  lot={lot}  ATR={atr:.2f}  SL±{sl_offset:.2f}  TP±{tp_offset:.2f}")
        return

    tick  = mt5.symbol_info_tick(symbol)
    price = tick.ask if direction == 'BUY' else tick.bid

    if direction == 'BUY':
        sl         = price - sl_offset
        tp         = price + tp_offset
        order_type = mt5.ORDER_TYPE_BUY
    else:
        sl         = price + sl_offset
        tp         = price - tp_offset
        order_type = mt5.ORDER_TYPE_SELL

    req = {
        'action':       mt5.TRADE_ACTION_DEAL,
        'symbol':       symbol,
        'volume':       lot,
        'type':         order_type,
        'price':        price,
        'sl':           sl,
        'tp':           tp,
        'magic':        magic,
        'deviation':    deviation,
        'comment':      'correlated_executor',
        'type_time':    mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(req)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        _log('OPEN', symbol,
             direction = direction,
             price     = price,
             sl        = round(sl, 5),
             tp_target = round(tp, 5),
             atr       = round(atr, 5))
        notify(f"✓ {symbol} {direction} @ {price:.2f}  SL={sl:.2f}  TP={tp:.2f}")
    else:
        _log('OPEN_FAILED', symbol,
             direction = direction,
             reason    = str(result.comment))
        notify(f"✗ {symbol} {direction} FAILED: {result.comment}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Symbol / execution
    parser.add_argument('--symbols',    default='NAS100,SP500,DOW30',
                        help='Comma-separated list of symbols to monitor')
    parser.add_argument('--lot',        type=float, default=1.0,
                        help='Lot size for every trade')
    parser.add_argument('--interval',   type=int,   default=5,
                        help='Main loop interval in seconds')

    # Risk management
    parser.add_argument('--sl_mult',    type=float, default=1.5,
                        help='Stop-loss distance = ATR × sl_mult')
    parser.add_argument('--tp_mult',    type=float, default=2.0,
                        help='Take-profit distance = ATR × tp_mult')

    # Informational / forwarded to inference process
    parser.add_argument('--atr_period', type=int,   default=14,
                        help='ATR period used by the upstream inference process '
                             '(informational — not used directly here)')

    # MT5 order parameters
    parser.add_argument('--magic',      type=int,   default=int(time.time()),
                        help='Magic number for orders placed by this script')
    parser.add_argument('--deviation',  type=int,   default=20,
                        help='Maximum price deviation (slippage) in points')

    # Inference staleness
    parser.add_argument('--max_age',    type=int,   default=30,
                        help='Reject inference rows older than this many seconds '
                             '(inference cycle ≈ 5 s per symbol; 30 s gives 6 cycles of headroom)')

    # Modes
    parser.add_argument('--dry_run',    action='store_true',
                        help='Log and notify without sending real orders')

    args    = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(',')]

    # Initialise per-symbol state
    states: Dict[str, SymbolState] = {sym: SymbolState() for sym in symbols}

    print(f"{c(Colors.CYAN, '=== Correlated Executor Started ===')}  dry_run={args.dry_run}")
    print(f"Symbols  : {', '.join(symbols)}")
    print(f"SL mult  : {args.sl_mult}  |  TP mult  : {args.tp_mult}  |  ATR period: {args.atr_period}")
    print(f"Max age  : {args.max_age}s  |  Interval : {args.interval}s")
    print(f"Reading  : {_INFERENCE_LOG_FILE}")

    try:
        while True:
            try:
                inferences = read_latest_inferences(symbols, max_age_sec=args.max_age)

                # Not all symbols have a fresh row yet → keep waiting
                if len(inferences) < len(symbols):
                    missing = set(symbols) - set(inferences.keys())
                    print(f"{c(Colors.YELLOW, '[WAITING]')} Stale/missing: {missing}")
                    time.sleep(args.interval)
                    continue

                # Update streak counters before scoring so consec_norm is live
                update_states(states, inferences)

                aligned, alignment = check_alignment(inferences)

                # Pre-compute scores for logging (select_best_symbol repeats this cheaply)
                scores = {
                    sym: calculate_score(inf, states[sym].consecutive_count)
                    for sym, inf in inferences.items()
                }

                # Write per-symbol inference snapshot with correlation context
                _log_inference(inferences, scores, states, alignment)

                best = select_best_symbol(inferences, states)

                if best:
                    symbol, score, alignment = best
                    inf = inferences[symbol]
                    print(
                        f"{c(Colors.GREEN, f'[{alignment}]')} {symbol}: {inf.signal_decision} "
                        f"score={score:.4f}  consec={states[symbol].consecutive_count}"
                    )
                    
                    # Log full correlation state snapshot
                    _log_correlation_state(inferences, scores, alignment, symbol, score, 'EXECUTE', symbols)

                    if not get_open_position(symbol):
                        open_position(
                            symbol    = symbol,
                            direction = inf.signal_decision,
                            lot       = args.lot,
                            sl_mult   = args.sl_mult,
                            tp_mult   = args.tp_mult,
                            atr       = inf.atr_value,
                            magic     = args.magic,
                            deviation = args.deviation,
                            dry_run   = args.dry_run,
                        )
                    else:
                        print(f"{c(Colors.YELLOW, '[SKIP]')} {symbol} already has an open position")
                else:
                    # Log correlation state even when not aligned
                    _log_correlation_state(inferences, scores, alignment, None, 0.0, 'SKIP', symbols)
                    print(f"{c(Colors.YELLOW, f'[{alignment}]')} Signals not aligned — waiting")

                time.sleep(args.interval)

            except Exception as e:
                print(f"{c(Colors.RED, f'[ERROR] {e}')}")
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\n{c(Colors.CYAN, '=== Correlated Executor Stopped ===')}")


if __name__ == '__main__':
    main()
