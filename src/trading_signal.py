"""
Produce an actionable trading signal from M5/H1 JSON trading data.

Usage:
    python trading_signal.py --input eurusd_data.json
    python trading_signal.py --input nq_data.json --output signal.json
    
    # Or pipe from another script:
    cat data.json | python trading_signal.py --stdin
"""

import argparse
import json
import sys


def safe_get(d, *keys, default=0.0):
    """Safely navigate nested dicts."""
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d


def compute_signal_score(data):
    """
    Compute a composite signal score (-1.0 to +1.0) and confidence (0.0 to 1.0)
    based on all available indicators across timeframes.
    """
    score = 0.0
    weights = []
    
    # ===== 5M INDICATORS =====
    m5 = safe_get(data, "timeframes", "5m", "indicators", default={})
    
    # WVO (Weighted Volume Oscillator)
    wvo = safe_get(m5, "WVO", default=0.0)
    if wvo > 0.0005:
        score += 0.15
        weights.append(("5m WVO bullish", 0.15))
    elif wvo < -0.0005:
        score -= 0.15
        weights.append(("5m WVO bearish", -0.15))
    
    # ARSI (Adaptive RSI)
    arsi = safe_get(m5, "ARSI", default=50.0)
    if arsi > 70:
        score -= 0.20
        weights.append(("5m ARSI overbought", -0.20))
    elif arsi < 30:
        score += 0.20
        weights.append(("5m ARSI oversold", 0.20))
    elif arsi > 55:
        score += 0.10
        weights.append(("5m ARSI bullish", 0.10))
    elif arsi < 45:
        score -= 0.10
        weights.append(("5m ARSI bearish", -0.10))
    
    # VWIO (Volume-Weighted Intraday Oscillator)
    vwio = safe_get(m5, "VWIO", default=0.0)
    if vwio > 5:
        score -= 0.10
        weights.append(("5m VWIO overextended", -0.10))
    elif vwio < -5:
        score += 0.10
        weights.append(("5m VWIO oversold", 0.10))
    
    # ===== 1H INDICATORS =====
    h1 = safe_get(data, "timeframes", "1h", "indicators", default={})
    
    # MACD Histogram
    macd_hist = safe_get(h1, "MACD_Histogram", default=0.0)
    if macd_hist > 5:
        score += 0.25
        weights.append(("1h MACD bullish", 0.25))
    elif macd_hist > 0:
        score += 0.10
        weights.append(("1h MACD weak bullish", 0.10))
    elif macd_hist < -5:
        score -= 0.25
        weights.append(("1h MACD bearish", -0.25))
    elif macd_hist < 0:
        score -= 0.10
        weights.append(("1h MACD weak bearish", -0.10))
    
    # Parabolic SAR vs Price
    psar = safe_get(h1, "Parabolic_SAR", default=0.0)
    h1_close = safe_get(data, "timeframes", "1h", "price", "close", default=0.0)
    if psar > 0 and h1_close > 0:
        if psar > h1_close:
            score -= 0.20
            weights.append(("1h PSAR bearish", -0.20))
        else:
            score += 0.20
            weights.append(("1h PSAR bullish", 0.20))
    
    # EMA 50/200 Crossover
    ema_cross = safe_get(h1, "EMA_50_200_Crossover", default="Neutral")
    if ema_cross == "Bullish":
        score += 0.20
        weights.append(("1h EMA crossover bullish", 0.20))
    elif ema_cross == "Bearish":
        score -= 0.20
        weights.append(("1h EMA crossover bearish", -0.20))
    
    # Hourly High-Low Percentile
    hl_pct = safe_get(h1, "Hourly_High_Low_Percentile", default=0.5)
    if hl_pct > 0.80:
        score -= 0.15
        weights.append(("1h near highs", -0.15))
    elif hl_pct < 0.20:
        score += 0.15
        weights.append(("1h near lows", 0.15))
    
    # Hourly Volume Momentum
    vol_mom = safe_get(h1, "Hourly_Volume_Momentum", default=1.0)
    if vol_mom > 1.5:
        if score > 0:
            score += 0.10
            weights.append(("1h volume confirms bullish", 0.10))
        elif score < 0:
            score -= 0.10
            weights.append(("1h volume confirms bearish", -0.10))
    
    # ===== SENTIMENT / MACRO =====
    sentiment = safe_get(data, "sentiment", default={})
    
    funding = safe_get(sentiment, "funding_rate", default=0.0)
    if funding < -0.01:
        score += 0.10
        weights.append(("funding bearish (shorts dominant)", 0.10))
    elif funding > 0.01:
        score -= 0.10
        weights.append(("funding bullish (longs dominant)", -0.10))
    
    fgi = safe_get(sentiment, "fear_greed_index", default=50)
    if fgi < 25:
        score += 0.10
        weights.append(("extreme fear", 0.10))
    elif fgi > 75:
        score -= 0.10
        weights.append(("extreme greed", -0.10))
    
    macro = safe_get(data, "macro_factors", default={})
    fomc = safe_get(macro, "fomc_event_impact", default="Neutral")
    if fomc == "Hawkish":
        score -= 0.10
        weights.append(("hawkish macro", -0.10))
    elif fomc == "Dovish":
        score += 0.10
        weights.append(("dovish macro", 0.10))
    
    # ===== CONFIDENCE =====
    total_weight = sum(abs(w[1]) for w in weights)
    if total_weight == 0:
        confidence = 0.50
    else:
        normalized = abs(score) / total_weight
        confidence = 0.50 + normalized * 0.45
    
    # Penalize conflicting signals
    bullish_count = sum(1 for _, w in weights if w > 0)
    bearish_count = sum(1 for _, w in weights if w < 0)
    if bullish_count > 0 and bearish_count > 0:
        confidence -= min(bullish_count, bearish_count) * 0.05
    
    confidence = max(0.50, min(0.95, confidence))
    
    return score, confidence, weights


def determine_action(score):
    """Convert score to BUY/SELL/HOLD action."""
    if score > 0.15:
        return "BUY"
    elif score < -0.15:
        return "SELL"
    else:
        return "HOLD"


def calculate_levels(data, action, score):
    """
    Calculate stop loss and take profit based on ATR, recent highs/lows,
    and a minimum Risk/Reward target of 1:1.5.
    """
    m5_price = safe_get(data, "timeframes", "5m", "price", default={})
    h1_price = safe_get(data, "timeframes", "1h", "price", default={})
    
    close = safe_get(m5_price, "close", default=0.0)
    if close == 0:
        close = safe_get(h1_price, "close", default=0.0)
    
    high = safe_get(m5_price, "high", default=close)
    low = safe_get(m5_price, "low", default=close)
    
    # Try to get ATR, else estimate from recent range
    atr = safe_get(data, "timeframes", "5m", "indicators", "ATR_14", default=0.0)
    if atr == 0:
        atr = (high - low) * 2
    
    # Asset-specific multipliers
    symbol = safe_get(data, "symbol", default="")
    is_fx = any(s in symbol for s in ["EUR", "GBP", "USD", "JPY", "AUD", "CAD", "CHF"])
    is_crypto = any(s in symbol for s in ["BTC", "ETH", "USDT"])
    
    if is_fx:
        atr_sl_mult, atr_tp_mult = 1.0, 2.0
    elif is_crypto:
        atr_sl_mult, atr_tp_mult = 2.0, 4.0
    else:
        atr_sl_mult, atr_tp_mult = 1.5, 3.0
    
    if action == "BUY":
        stop_loss = close - (atr * atr_sl_mult)
        take_profit = close + (atr * atr_tp_mult)
    elif action == "SELL":
        stop_loss = close + (atr * atr_sl_mult)
        take_profit = close - (atr * atr_tp_mult)
    else:
        stop_loss = close - (atr * 1.0)
        take_profit = close + (atr * 1.0)
    
    # Round decimals
    decimals = 5 if is_fx else (2 if is_crypto or close > 100 else 5)
    stop_loss = round(stop_loss, decimals)
    take_profit = round(take_profit, decimals)
    
    # Enforce minimum 1:1.5 R:R
    risk = abs(close - stop_loss)
    reward = abs(take_profit - close)
    if action in ("BUY", "SELL") and risk > 0 and reward / risk < 1.5:
        if action == "BUY":
            take_profit = round(close + risk * 1.5, decimals)
        else:
            take_profit = round(close - risk * 1.5, decimals)
    
    return stop_loss, take_profit


def generate_signal(data):
    """Main function: generate the trading signal JSON."""
    score, confidence, weights = compute_signal_score(data)
    action = determine_action(score)
    stop_loss, take_profit = calculate_levels(data, action, score)
    
    result = {
        "symbol": safe_get(data, "symbol", default="UNKNOWN"),
        "timestamp": safe_get(data, "timestamp", default=0),
        "prediction": {
            "action": action,
            "confidence": round(confidence, 2),
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }
    }
    
    # Optional debug info
    result["_debug"] = {
        "score": round(score, 4),
        "weights": weights,
        "rationale": generate_rationale(weights, action)
    }
    
    return result


def generate_rationale(weights, action):
    """Generate a human-readable rationale string."""
    bullish = [w[0] for w in weights if w[1] > 0]
    bearish = [w[0] for w in weights if w[1] < 0]
    
    parts = []
    if action == "BUY":
        parts.append(f"Bullish signals: {', '.join(bullish)}")
        if bearish:
            parts.append(f"Counter signals: {', '.join(bearish)}")
    elif action == "SELL":
        parts.append(f"Bearish signals: {', '.join(bearish)}")
        if bullish:
            parts.append(f"Counter signals: {', '.join(bullish)}")
    else:
        parts.append("Mixed signals — no clear directional bias.")
    
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate trading signal from JSON data")
    parser.add_argument("--input", "-i", help="Path to input JSON file")
    parser.add_argument("--output", "-o", help="Path to output JSON file")
    parser.add_argument("--stdin", action="store_true", help="Read JSON from stdin")
    parser.add_argument("--no-debug", action="store_true", help="Remove debug info from output")
    args = parser.parse_args()
    
    if args.stdin or (not args.input):
        raw = sys.stdin.read()
    else:
        with open(args.input, "r") as f:
            raw = f.read()
    
    if not raw.strip():
        print("Error: No input data provided.", file=sys.stderr)
        sys.exit(1)
    
    data = json.loads(raw)
    signal = generate_signal(data)
    
    if args.no_debug:
        signal.pop("_debug", None)
    
    output = json.dumps(signal, indent=2)
    print(output)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\nSaved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
