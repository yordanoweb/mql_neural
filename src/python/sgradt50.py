import MetaTrader5 as mt5
import onnxruntime as ort
import numpy as np
import pandas as pd
import time
import argparse
from datetime import datetime

# === IMPORTS DE LA LIBRERÍA TA ===
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import ATRIndicator
from ta.momentum import StochasticOscillator

# ====================== PARÁMETROS POR DEFECTO (iguales que en MQL5) ======================
InpModelName        = "EUR_USD_H1_SGRADT50.onnx"
InpMinConf          = 0.55
InpWindowSize       = 20
InpFeaturesPerBar   = 5
InpReverseInference = False

InpInferSeconds     = 15
InpOneTradePerBar   = True

InpStartHour        = 0
InpEndHour          = 24

InpEMAPeriod        = 9

InpStochK           = 7
InpStochD           = 3
InpStochSlowing     = 3
InpADXPeriod        = 8

InpLot              = 1.0
InpMagic            = 5050
InpStopPoints       = 50.0
InpTakePoints       = 100.0

InpUseATR           = False
InpATRPeriod        = 14
InpATRSLMultiplier  = 1.5
InpATRTPMultiplier  = 3.0

InpShowPanel        = True

# Valores por defecto para symbol y timeframe
symbol_default      = "EURUSD"
timeframe_default   = "H1"

# ====================== VARIABLES GLOBALES ======================
sess = None
g_last_infer       = 0
g_last_traded_bar  = 0
g_infer_count      = 0
g_prediction       = 0
g_conf_hold        = 0.0
g_conf_buy         = 0.0
g_conf_sell        = 0.0
g_curr_adx         = 0.0
g_curr_pdi         = 0.0
g_curr_mdi         = 0.0
g_stoch_k          = 0.0
g_stoch_d          = 0.0

# ====================== HELPERS ======================
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_timeframe(tf_str):
    tf_map = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
    return tf_map.get(tf_str.upper(), mt5.TIMEFRAME_H1)

# ====================== PREPARAR FEATURES ======================
def prepare_features():
    global symbol, timeframe
    window = InpWindowSize
    extra = 200
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, window + extra)
    if rates is None or len(rates) < window + 50:
        print("❌ ERROR: No se pudieron obtener rates")
        return None

    df = pd.DataFrame(rates)
    high = df['high']
    low = df['low']
    close = df['close']

    adx_ind = ADXIndicator(high=high, low=low, close=close, window=InpADXPeriod)
    stoch = StochasticOscillator(high=high, low=low, close=close,
                                 window=InpStochK, smooth_window=InpStochSlowing)

    adx_data = adx_ind.adx().iloc[-window:].values
    pdi_data = adx_ind.adx_pos().iloc[-window:].values
    mdi_data = adx_ind.adx_neg().iloc[-window:].values
    stoch_k_data = stoch.stoch().iloc[-window:].values
    stoch_d_data = stoch.stoch_signal().iloc[-window:].values

    total_features = window * InpFeaturesPerBar
    features = np.zeros(total_features, dtype=np.float32)

    for i in range(window):
        idx = i * InpFeaturesPerBar
        features[idx + 0] = stoch_k_data[i]
        features[idx + 1] = stoch_d_data[i]
        features[idx + 2] = adx_data[i]
        features[idx + 3] = pdi_data[i]
        features[idx + 4] = mdi_data[i]

    return features

# ====================== INDICADORES PARA PANEL ======================
def update_indicator_globals():
    global g_curr_adx, g_curr_pdi, g_curr_mdi, g_stoch_k, g_stoch_d, symbol, timeframe
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 300)
    if len(rates) < 100:
        return
    df = pd.DataFrame(rates)
    high = df['high']
    low = df['low']
    close = df['close']

    adx_ind = ADXIndicator(high=high, low=low, close=close, window=InpADXPeriod)
    stoch = StochasticOscillator(high=high, low=low, close=close,
                                 window=InpStochK, smooth_window=InpStochSlowing)

    g_curr_adx = adx_ind.adx().iloc[-1]
    g_curr_pdi = adx_ind.adx_pos().iloc[-1]
    g_curr_mdi = adx_ind.adx_neg().iloc[-1]
    g_stoch_k = stoch.stoch().iloc[-1]
    g_stoch_d = stoch.stoch_signal().iloc[-1]

# ====================== FILTRO EMA ======================
def ema_gate_allows(predicted_class):
    global symbol, timeframe
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, InpEMAPeriod + 50)
    if len(rates) < InpEMAPeriod + 5:
        return False
    close = pd.Series([r['close'] for r in rates])
    ema = EMAIndicator(close=close, window=InpEMAPeriod).ema_indicator().iloc[-1]

    tick = mt5.symbol_info_tick(symbol)
    if predicted_class == 1:
        return tick.ask > ema
    if predicted_class == 2:
        return tick.bid < ema
    return False

# ====================== SL / TP ======================
def get_sltp_distance():
    global symbol, timeframe
    point = mt5.symbol_info(symbol).point
    digits = mt5.symbol_info(symbol).digits

    if InpUseATR:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, InpATRPeriod + 50)
        df = pd.DataFrame(rates)
        atr = ATRIndicator(high=df['high'], low=df['low'], close=df['close'],
                           window=InpATRPeriod).atr()
        atr_value = atr.iloc[-2]
        sl_dist = atr_value * InpATRSLMultiplier if InpATRSLMultiplier > 0 else 0
        tp_dist = atr_value * InpATRTPMultiplier if InpATRTPMultiplier > 0 else 0
        print(f"ATR: {atr_value:.5f} | SL dist: {sl_dist:.5f} | TP dist: {tp_dist:.5f}")
        return True, sl_dist, tp_dist
    else:
        sl_dist = InpStopPoints * point
        tp_dist = InpTakePoints * point
        return True, sl_dist, tp_dist

# ====================== INFERENCIA ONNX ======================
def run_inference():
    global g_last_infer, g_infer_count, g_prediction, g_conf_hold, g_conf_buy, g_conf_sell, g_last_traded_bar, symbol, timeframe

    g_last_infer = time.time()
    g_infer_count += 1

    features = prepare_features()
    if features is None:
        return

    input_data = features.reshape(1, -1).astype(np.float32)

    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    outputs = sess.run(output_names, {input_name: input_data})

    g_prediction = int(outputs[0].item())
    probs = outputs[1][0]
    g_conf_hold = float(probs[0])
    g_conf_buy  = float(probs[1])
    g_conf_sell = float(probs[2])

    if InpReverseInference:
        if g_prediction == 1: g_prediction = 2
        elif g_prediction == 2: g_prediction = 1

    active_conf = g_conf_buy if g_prediction == 1 else g_conf_sell if g_prediction == 2 else 0.0

    tick = mt5.symbol_info_tick(symbol)
    dt = datetime.fromtimestamp(tick.time) if tick else datetime.now()
    time_ok = InpStartHour <= dt.hour < InpEndHour

    rates1 = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
    current_bar = rates1[0]['time'] if len(rates1) > 0 else 0
    bar_ok = (current_bar != g_last_traded_bar) if InpOneTradePerBar else True

    positions = mt5.positions_get(symbol=symbol)
    no_position = len(positions) == 0 if positions is not None else True

    signal_name = "HOLD" if g_prediction == 0 else "BUY" if g_prediction == 1 else "SELL"
    display_conf = g_conf_hold if g_prediction == 0 else active_conf

    print(f"\n{symbol} | Inference #{g_infer_count}: {signal_name} | Conf: {display_conf*100:.2f}% | "
          f"Time: {'OK' if time_ok else 'CLOSED'} | Bar: {'OK' if bar_ok else 'SKIP'} | Pos: {'NONE' if no_position else 'OPEN'}")

    if (g_prediction > 0 and active_conf >= InpMinConf and time_ok and bar_ok and no_position and
        ema_gate_allows(g_prediction)):

        success, sl_dist, tp_dist = get_sltp_distance()
        if not success:
            return

        digits = mt5.symbol_info(symbol).digits

        if g_prediction == 1:  # BUY
            ask = tick.ask
            sl = round(ask - sl_dist, digits)
            tp = round(ask + tp_dist, digits)
            comment = f"SGRADT50 BUY @{g_conf_buy*100:.1f}%"

            request = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": InpLot,
                "type": mt5.ORDER_TYPE_BUY, "price": ask, "sl": sl, "tp": tp,
                "deviation": 10, "magic": InpMagic, "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK,
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                g_last_traded_bar = current_bar
                print("✅ BUY ejecutada correctamente")
            else:
                print(f"❌ BUY falló: {result.retcode} - {result.comment}")

        elif g_prediction == 2:  # SELL
            bid = tick.bid
            sl = round(bid + sl_dist, digits)
            tp = round(bid - tp_dist, digits)
            comment = f"SGRADT50 SELL @{g_conf_sell*100:.1f}%"

            request = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": InpLot,
                "type": mt5.ORDER_TYPE_SELL, "price": bid, "sl": sl, "tp": tp,
                "deviation": 10, "magic": InpMagic, "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK,
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                g_last_traded_bar = current_bar
                print("✅ SELL ejecutada correctamente")
            else:
                print(f"❌ SELL falló: {result.retcode} - {result.comment}")

    elif g_prediction > 0:
        if active_conf < InpMinConf:
            print(f"Signal rechazado: Confianza {active_conf*100:.2f}% < {InpMinConf*100:.1f}%")
        if not time_ok: print("Signal rechazado: Fuera de horario")
        if not bar_ok: print("Signal rechazado: Ya se operó esta barra")
        if not no_position: print("Signal rechazado: Ya hay posición abierta")
        if not ema_gate_allows(g_prediction):
            print(f"Signal rechazado: Entrada equivocada respecto a EMA({InpEMAPeriod})")

# ====================== PANEL ======================
def show_status():
    global symbol
    tick = mt5.symbol_info_tick(symbol)
    dt = datetime.fromtimestamp(tick.time) if tick else datetime.now()
    valid_time = InpStartHour <= dt.hour < InpEndHour

    signal_text = "HOLD" if g_prediction == 0 else "BUY" if g_prediction == 1 else "SELL"

    info = "\n" + "="*70 + "\n"
    info += f"SESSION: {InpStartHour:02d}:00-{InpEndHour:02d} [{'ACTIVE' if valid_time else 'CLOSED'}]\n"
    info += f"MODE: {'TIMER '+str(InpInferSeconds)+'s' if InpInferSeconds > 0 else 'NEW BAR'} | Runs: {g_infer_count}\n"
    if g_last_infer > 0:
        info += f"Last inference: {datetime.fromtimestamp(g_last_infer).strftime('%H:%M:%S')}\n"
    info += "-"*70 + "\n"
    info += f"INDICATORS\nADX: {g_curr_adx:.1f} | +DI: {g_curr_pdi:.1f} | -DI: {g_curr_mdi:.1f}\n"
    info += f"Stoch K: {g_stoch_k:.1f} | D: {g_stoch_d:.1f}\n"
    info += "-"*70 + "\n"
    info += f"AI SIGNAL: {signal_text}\n"
    info += f"HOLD: {g_conf_hold*100:.1f}% | BUY: {g_conf_buy*100:.1f}% | SELL: {g_conf_sell*100:.1f}%\n"
    info += f"Min Conf: {InpMinConf*100:.1f}%\n"
    info += "-"*70 + "\n"
    info += f"LOT: {InpLot:.2f} | "
    if InpUseATR:
        info += f"SL/TP: ATR({InpATRPeriod}) x{InpATRSLMultiplier:.1f}/{InpATRTPMultiplier:.1f}\n"
    else:
        info += f"SL: {InpStopPoints:.0f} | TP: {InpTakePoints:.0f}\n"

    positions = mt5.positions_get(symbol=symbol)
    if positions:
        pos = positions[0]
        type_str = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
        info += f"POSITION: {type_str} | PL: {pos.profit:+.2f} {mt5.account_info().currency}\n"
    else:
        info += "Waiting for signal...\n"
    info += "="*70
    print(info)

# ====================== CICLO PRINCIPAL ======================
def main():
    global sess, symbol, timeframe
    global InpModelName, InpMinConf, InpWindowSize, InpFeaturesPerBar, InpReverseInference
    global InpInferSeconds, InpOneTradePerBar, InpStartHour, InpEndHour, InpEMAPeriod
    global InpStochK, InpStochD, InpStochSlowing, InpADXPeriod
    global InpLot, InpMagic, InpStopPoints, InpTakePoints
    global InpUseATR, InpATRPeriod, InpATRSLMultiplier, InpATRTPMultiplier
    global InpShowPanel

    # ====================== ARGUMENTOS DE LÍNEA DE COMANDOS ======================
    parser = argparse.ArgumentParser(
        description='SGRADT 5.0 - Python Edition'
    )
    parser.add_argument('--model', type=str, default=InpModelName,
                        help='Archivo ONNX')
    parser.add_argument('--min-conf', type=float, default=InpMinConf,
                        help='Minimum confidence threshold')
    parser.add_argument('--window-size', type=int, default=InpWindowSize,
                        help='Window size')
    parser.add_argument('--features-per-bar', type=int, default=InpFeaturesPerBar,
                        help='Features per bar')
    parser.add_argument('--reverse-inference', type=str2bool, default=InpReverseInference,
                        help='Reverse the inference signal')
    parser.add_argument('--infer-seconds', type=int, default=InpInferSeconds,
                        help='Run inference every N seconds (0 = new bar only)')
    parser.add_argument('--one-trade-per-bar', type=str2bool, default=InpOneTradePerBar,
                        help='Limit to 1 trade per bar')
    parser.add_argument('--start-hour', type=int, default=InpStartHour,
                        help='Session start hour')
    parser.add_argument('--end-hour', type=int, default=InpEndHour,
                        help='Session end hour')
    parser.add_argument('--ema-period', type=int, default=InpEMAPeriod,
                        help='EMA period')
    parser.add_argument('--stoch-k', type=int, default=InpStochK,
                        help='Stochastic %K period')
    parser.add_argument('--stoch-d', type=int, default=InpStochD,
                        help='Stochastic %D period')
    parser.add_argument('--stoch-slowing', type=int, default=InpStochSlowing,
                        help='Stochastic slowing')
    parser.add_argument('--adx-period', type=int, default=InpADXPeriod,
                        help='ADX period')
    parser.add_argument('--lot', type=float, default=InpLot,
                        help='Lot size')
    parser.add_argument('--magic', type=int, default=InpMagic,
                        help='Magic number')
    parser.add_argument('--stop-points', type=float, default=InpStopPoints,
                        help='SL points')
    parser.add_argument('--take-points', type=float, default=InpTakePoints,
                        help='TP points')
    parser.add_argument('--use-atr', type=str2bool, default=InpUseATR,
                        help='Use ATR-based SL/TP')
    parser.add_argument('--atr-period', type=int, default=InpATRPeriod,
                        help='ATR period')
    parser.add_argument('--atr-sl-multiplier', type=float, default=InpATRSLMultiplier,
                        help='ATR SL multiplier')
    parser.add_argument('--atr-tp-multiplier', type=float, default=InpATRTPMultiplier,
                        help='ATR TP multiplier')
    parser.add_argument('--show-panel', type=str2bool, default=InpShowPanel,
                        help='Show information panel')

    # === SYMBOL Y TIMEFRAME (también por línea de comandos) ===
    parser.add_argument('--symbol', type=str, default=symbol_default,
                        help='Trading symbol (ej: EURUSD, GBPUSD, BTCUSD)')
    parser.add_argument('--timeframe', type=str, default=timeframe_default,
                        help='Timeframe (M1, M5, M15, H1, H4, D1, W1, MN1)')

    args = parser.parse_args()

    # Aplicar todos los valores de la línea de comandos
    symbol                  = args.symbol
    timeframe               = parse_timeframe(args.timeframe)
    InpModelName            = args.model
    InpMinConf              = args.min_conf
    InpWindowSize           = args.window_size
    InpFeaturesPerBar       = args.features_per_bar
    InpReverseInference     = args.reverse_inference
    InpInferSeconds         = args.infer_seconds
    InpOneTradePerBar       = args.one_trade_per_bar
    InpStartHour            = args.start_hour
    InpEndHour              = args.end_hour
    InpEMAPeriod            = args.ema_period
    InpStochK               = args.stoch_k
    InpStochD               = args.stoch_d
    InpStochSlowing         = args.stoch_slowing
    InpADXPeriod            = args.adx_period
    InpLot                  = args.lot
    InpMagic                = args.magic
    InpStopPoints           = args.stop_points
    InpTakePoints           = args.take_points
    InpUseATR               = args.use_atr
    InpATRPeriod            = args.atr_period
    InpATRSLMultiplier      = args.atr_sl_multiplier
    InpATRTPMultiplier      = args.atr_tp_multiplier
    InpShowPanel            = args.show_panel

    # ====================== INICIALIZACIÓN ======================
    if not mt5.initialize():
        print("❌ MT5 initialize() failed")
        return

    print(f"✅ MetaTrader5 conectado | Symbol: {symbol} | Timeframe: {args.timeframe}")
    if not mt5.symbol_select(symbol, True):
        print(f"❌ No se pudo seleccionar {symbol}")

    try:
        sess = ort.InferenceSession(InpModelName)
        print(f"✅ ONNX model loaded: {InpModelName}")
    except Exception as e:
        print(f"❌ Error cargando ONNX: {e}")
        mt5.shutdown()
        return

    print("✅ EA_SGRADT50_ONNX (Python) iniciado correctamente\n")

    last_bar_time = 0
    last_infer_time = 0.0

    while True:
        update_indicator_globals()

        run_now = False
        rates1 = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
        current_bar = rates1[0]['time'] if len(rates1) > 0 else 0

        if InpInferSeconds <= 0:
            if current_bar != last_bar_time and current_bar != 0:
                last_bar_time = current_bar
                run_now = True
        else:
            now = time.time()
            if now - last_infer_time >= InpInferSeconds:
                last_infer_time = now
                run_now = True

        if run_now:
            run_inference()

        if InpShowPanel:
            show_status()

        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ EA detenido por el usuario")
    finally:
        if mt5.terminal_info():
            mt5.shutdown()
