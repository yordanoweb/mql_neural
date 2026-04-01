"""
Script de Verificación: Sincronización Python - MetaTrader 5
Verifica que las features calculadas sean idénticas en ambos sistemas
"""

import pandas as pd
import numpy as np
import ta
from colorama import Fore, Style, init

init(autoreset=True)

def calculate_stochastic_features(high, low, close, window=14, smooth_k=3, smooth_d=3):
    """Exact same calculation as in training script"""
    stoch = ta.momentum.StochasticOscillator(high, low, close, 
                                             window=window, 
                                             smooth_window=smooth_k)
    
    stoch_k = stoch.stoch()
    stoch_d = stoch.stoch_signal()
    
    feat_stoch_momentum = (stoch_k - stoch_d) / 100.0
    feat_stoch_position = (stoch_k - 50.0) / 50.0
    feat_stoch_velocity = stoch_k.diff() / 100.0
    
    overbought_pressure = np.where(stoch_k > 80, -(stoch_k - 80) / 20.0, 0)
    oversold_pressure = np.where(stoch_k < 20, (20 - stoch_k) / 20.0, 0)
    feat_stoch_divergence = overbought_pressure + oversold_pressure
    
    return {
        'stoch_k': stoch_k,
        'stoch_d': stoch_d,
        'feat_stoch_momentum': feat_stoch_momentum,
        'feat_stoch_position': feat_stoch_position,
        'feat_stoch_velocity': feat_stoch_velocity,
        'feat_stoch_divergence': feat_stoch_divergence
    }

def calculate_volume_features(tick_volume, close, window=20):
    """Exact same calculation as in training script"""
    vol_ma = tick_volume.rolling(window=window).mean()
    vol_std = tick_volume.rolling(window=window).std()
    
    feat_vol_ratio = tick_volume / vol_ma.replace(0, 1)
    
    vol_ema_fast = tick_volume.ewm(span=5, adjust=False).mean()
    vol_ema_slow = tick_volume.ewm(span=20, adjust=False).mean()
    feat_vol_momentum = (vol_ema_fast - vol_ema_slow) / vol_ema_slow.replace(0, 1)
    
    price_change = close.pct_change().abs()
    vol_change = tick_volume.pct_change().abs()
    feat_vol_price_div = (vol_change - price_change).fillna(0)
    
    feat_vol_percentile = tick_volume.rolling(window=window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    )
    feat_vol_percentile = (feat_vol_percentile - 0.5) * 2
    
    feat_vol_zscore = (tick_volume - vol_ma) / vol_std.replace(0, 1)
    feat_vol_zscore = np.clip(feat_vol_zscore, -3, 3) / 3.0
    
    return {
        'vol_ma': vol_ma,
        'vol_std': vol_std,
        'feat_vol_ratio': feat_vol_ratio,
        'feat_vol_momentum': feat_vol_momentum,
        'feat_vol_price_div': feat_vol_price_div,
        'feat_vol_percentile': feat_vol_percentile,
        'feat_vol_zscore': feat_vol_zscore
    }

def verify_features(csv_file, window=20, atr_period=14, stoch_window=14, vol_window=20):
    """Verify feature calculations and show examples"""
    
    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.CYAN}VERIFICACIÓN DE SINCRONIZACIÓN PYTHON - MT5")
    print(f"{Fore.CYAN}{'=' * 80}\n")
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"{Fore.GREEN}✓ Datos cargados: {len(df)} filas\n")
    
    # Calculate ATR
    atr_indicator = ta.volatility.AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=atr_period
    )
    df['atr'] = atr_indicator.average_true_range()
    
    # Basic features
    df['feat_body'] = (df['close'] - df['open']) / df['atr']
    df['feat_range'] = (df['high'] - df['low']) / df['atr']
    
    # Stochastic features
    stoch_features = calculate_stochastic_features(
        df['high'], df['low'], df['close'], 
        window=stoch_window
    )
    for key, value in stoch_features.items():
        df[key] = value
    
    # Volume features
    volume_features = calculate_volume_features(
        df['tick_volume'], df['close'], 
        window=vol_window
    )
    for key, value in volume_features.items():
        df[key] = value
    
    df.dropna(inplace=True)
    
    # Show feature order
    print(f"{Fore.YELLOW}ORDEN DE FEATURES (debe coincidir con MT5):")
    print(f"{Fore.YELLOW}{'-' * 80}")
    features = [
        'feat_body',
        'feat_range',
        'feat_stoch_momentum',
        'feat_stoch_position',
        'feat_stoch_velocity',
        'feat_stoch_divergence',
        'feat_vol_ratio',
        'feat_vol_momentum',
        'feat_vol_price_div',
        'feat_vol_percentile',
        'feat_vol_zscore'
    ]
    
    for i, feat in enumerate(features):
        print(f"{Fore.WHITE}  {i:2d}. {feat}")
    
    # Show recent examples
    print(f"\n{Fore.CYAN}EJEMPLOS DE FEATURES CALCULADAS (últimas 5 velas):")
    print(f"{Fore.CYAN}{'-' * 80}")
    
    recent = df[features].tail(5)
    
    print(f"\n{Fore.MAGENTA}BASIC FEATURES:")
    print(recent[['feat_body', 'feat_range']].to_string())
    
    print(f"\n{Fore.MAGENTA}STOCHASTIC FEATURES:")
    print(recent[['feat_stoch_momentum', 'feat_stoch_position', 
                  'feat_stoch_velocity', 'feat_stoch_divergence']].to_string())
    
    print(f"\n{Fore.MAGENTA}VOLUME FEATURES:")
    print(recent[['feat_vol_ratio', 'feat_vol_momentum', 'feat_vol_price_div',
                  'feat_vol_percentile', 'feat_vol_zscore']].to_string())
    
    # Statistical verification
    print(f"\n{Fore.CYAN}VERIFICACIÓN ESTADÍSTICA:")
    print(f"{Fore.CYAN}{'-' * 80}")
    
    for feat in features:
        mean = df[feat].mean()
        std = df[feat].std()
        min_val = df[feat].min()
        max_val = df[feat].max()
        
        # Check if normalized properly
        if abs(mean) > 2:
            status = f"{Fore.RED}⚠ REVISAR (media alta)"
        elif std > 5:
            status = f"{Fore.RED}⚠ REVISAR (std alta)"
        else:
            status = f"{Fore.GREEN}✓ OK"
        
        print(f"{feat:25s} → Mean: {mean:7.3f}, Std: {std:6.3f}, "
              f"Range: [{min_val:7.3f}, {max_val:7.3f}] {status}")
    
    # Generate MQL5 verification code
    print(f"\n{Fore.CYAN}CÓDIGO MQL5 PARA VERIFICACIÓN MANUAL:")
    print(f"{Fore.CYAN}{'-' * 80}")
    
    last_row = df.iloc[-1]
    
    print(f"{Fore.WHITE}// Use estos valores en MT5 para verificar manualmente:")
    print(f"{Fore.WHITE}// Última vela procesada:")
    print(f"{Fore.YELLOW}double test_close = {last_row['close']:.5f};")
    print(f"{Fore.YELLOW}double test_open = {last_row['open']:.5f};")
    print(f"{Fore.YELLOW}double test_high = {last_row['high']:.5f};")
    print(f"{Fore.YELLOW}double test_low = {last_row['low']:.5f};")
    print(f"{Fore.YELLOW}long test_volume = {int(last_row['tick_volume'])};")
    print(f"{Fore.YELLOW}double test_atr = {last_row['atr']:.5f};")
    print()
    print(f"{Fore.WHITE}// Features esperadas (Python):")
    for i, feat in enumerate(features):
        val = last_row[feat]
        print(f"{Fore.GREEN}// input_buffer[{i:2d}] debería ser ≈ {val:8.5f}  // {feat}")
    
    # Count check
    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.GREEN}✓ Total de features: {len(features)}")
    print(f"{Fore.GREEN}✓ Input shape esperado: [1, {window * len(features)}]")
    print(f"{Fore.GREEN}✓ Ventana: {window} barras")
    print(f"{Fore.CYAN}{'=' * 80}\n")
    
    return df, features

def generate_mt5_test_code(df, features, window=20):
    """Generate MQL5 code to manually test feature calculation"""
    
    print(f"\n{Fore.CYAN}CÓDIGO MQL5 COMPLETO PARA PRUEBA:")
    print(f"{Fore.CYAN}{'=' * 80}\n")
    
    code = f"""
//+------------------------------------------------------------------+
//| Script de verificación de features                              |
//+------------------------------------------------------------------+
void OnStart()
{{
   Print("=== VERIFICACIÓN DE FEATURES ===");
   
   // Obtener datos
   double close[], open[], high[], low[];
   long volumes[];
   
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(volumes, true);
   
   int bars = {window + 30};
   CopyClose(_Symbol, _Period, 0, bars, close);
   CopyOpen(_Symbol, _Period, 0, bars, open);
   CopyHigh(_Symbol, _Period, 0, bars, high);
   CopyLow(_Symbol, _Period, 0, bars, low);
   CopyTickVolume(_Symbol, _Period, 0, bars, volumes);
   
   // ATR
   int atr_handle = iATR(_Symbol, _Period, 14);
   double atr[];
   ArraySetAsSeries(atr, true);
   CopyBuffer(atr_handle, 0, 0, {window}, atr);
   
   // Stochastic
   int stoch_handle = iStochastic(_Symbol, _Period, 14, 3, 3, MODE_SMA, STO_LOWHIGH);
   double stoch_main[], stoch_signal[];
   ArraySetAsSeries(stoch_main, true);
   ArraySetAsSeries(stoch_signal, true);
   CopyBuffer(stoch_handle, MAIN_LINE, 0, {window + 1}, stoch_main);
   CopyBuffer(stoch_handle, SIGNAL_LINE, 0, {window + 1}, stoch_signal);
   
   // Calcular features para la barra más reciente (idx=0)
   int idx = 0;
   
   Print("\\n--- VALORES BASE ---");
   Print("Close: ", close[idx]);
   Print("Open: ", open[idx]);
   Print("High: ", high[idx]);
   Print("Low: ", low[idx]);
   Print("Volume: ", volumes[idx]);
   Print("ATR: ", atr[idx]);
   Print("Stoch K: ", stoch_main[idx]);
   Print("Stoch D: ", stoch_signal[idx]);
   
   Print("\\n--- FEATURES CALCULADAS ---");
   
   // Basic features
   float feat_body = (float)((close[idx] - open[idx]) / atr[idx]);
   float feat_range = (float)((high[idx] - low[idx]) / atr[idx]);
   Print("feat_body: ", feat_body);
   Print("feat_range: ", feat_range);
   
   // Stochastic features
   float stoch_momentum, stoch_position, stoch_velocity, stoch_divergence;
   double k = stoch_main[idx];
   double d = stoch_signal[idx];
   
   stoch_momentum = (float)((k - d) / 100.0);
   stoch_position = (float)((k - 50.0) / 50.0);
   stoch_velocity = (float)((k - stoch_main[idx+1]) / 100.0);
   
   float overbought = (k > 80) ? (float)(-(k - 80) / 20.0) : 0.0;
   float oversold = (k < 20) ? (float)((20 - k) / 20.0) : 0.0;
   stoch_divergence = overbought + oversold;
   
   Print("feat_stoch_momentum: ", stoch_momentum);
   Print("feat_stoch_position: ", stoch_position);
   Print("feat_stoch_velocity: ", stoch_velocity);
   Print("feat_stoch_divergence: ", stoch_divergence);
   
   // Volume features (simplificado)
   double vol_sum = 0;
   for(int i=0; i<20; i++) vol_sum += (double)volumes[idx + i];
   double vol_ma = vol_sum / 20.0;
   
   float vol_ratio = (float)((double)volumes[idx] / vol_ma);
   Print("feat_vol_ratio: ", vol_ratio);
   
   Print("\\n--- COMPARAR CON PYTHON ---");
   Print("Si los valores no coinciden, revisar:");
   Print("1. Misma ventana temporal");
   Print("2. Mismos parámetros de indicadores");
   Print("3. Mismo orden de datos (series o no series)");
   
   IndicatorRelease(atr_handle);
   IndicatorRelease(stoch_handle);
}}
"""
    
    print(Fore.YELLOW + code)
    print(f"{Fore.CYAN}{'=' * 80}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Python-MT5 feature synchronization")
    parser.add_argument("--csv", type=str, required=True, help="CSV file to verify")
    parser.add_argument("--window", type=int, default=20, help="Window size")
    parser.add_argument("--atr_period", type=int, default=14, help="ATR period")
    parser.add_argument("--stoch_window", type=int, default=14, help="Stochastic window")
    parser.add_argument("--vol_window", type=int, default=20, help="Volume window")
    
    args = parser.parse_args()
    
    df, features = verify_features(
        args.csv, 
        window=args.window,
        atr_period=args.atr_period,
        stoch_window=args.stoch_window,
        vol_window=args.vol_window
    )
    
    generate_mt5_test_code(df, features, window=args.window)
    
    print(f"{Fore.GREEN}✓ Verificación completada!")
    print(f"{Fore.YELLOW}Siguiente paso: Ejecutar el script MQL5 generado en MT5")
