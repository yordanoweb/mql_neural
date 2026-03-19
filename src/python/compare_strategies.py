"""
Script de Comparación: Original vs SGRADT 5.0
Compara las señales generadas por ambos métodos para visualizar diferencias.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator
import warnings

warnings.filterwarnings('ignore')


def calculate_original_signals(df, stoch_k=5, stoch_d=3, oversold=30, overbought=70, adx_thresh=24):
    """Método original del script train_price_action_adx_points.py"""
    
    # Stochastic
    stoch = StochasticOscillator(
        high=df['high'], low=df['low'], close=df['close'],
        window=stoch_k, smooth_window=stoch_d
    )
    k = stoch.stoch()
    d = stoch.stoch_signal()
    k_prev = k.shift(1)
    d_prev = d.shift(1)
    
    # Original: simple crossover from zone
    stoch_cross_up = (k_prev <= d_prev) & (k > d) & (k_prev < oversold)
    stoch_cross_down = (k_prev >= d_prev) & (k < d) & (k_prev > overbought)
    
    # ADX
    adx_inst = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    adx = adx_inst.adx()
    pdi = adx_inst.adx_pos()
    mdi = adx_inst.adx_neg()
    
    # Original: simple threshold check
    buy_signals = (adx > adx_thresh) & (pdi > mdi) & stoch_cross_up
    sell_signals = (adx > adx_thresh) & (mdi > pdi) & stoch_cross_down
    
    return {
        'buy': buy_signals,
        'sell': sell_signals,
        'stoch_k': k,
        'stoch_d': d,
        'adx': adx,
        'pdi': pdi,
        'mdi': mdi
    }


def calculate_sgradt_signals(df, stoch_k=7, stoch_d=3, oversold=20, overbought=80, 
                             adx_period=8, adx_limit=32):
    """Método SGRADT 5.0 con todas las condiciones del markdown"""
    
    # Stochastic
    stoch = StochasticOscillator(
        high=df['high'], low=df['low'], close=df['close'],
        window=stoch_k, smooth_window=stoch_d
    )
    main = stoch.stoch()
    signal = stoch.stoch_signal()
    
    # Referencias a barras
    main_0, main_1, main_2, main_3 = main, main.shift(1), main.shift(2), main.shift(3)
    signal_0, signal_1, signal_2, signal_3 = signal, signal.shift(1), signal.shift(2), signal.shift(3)
    
    # STOCHASTIC BUY
    buy_oversold_1 = (main_2 < signal_2) & (main_1 > signal_1) & (main_1 <= oversold)
    buy_oversold_2 = (main_3 < signal_3) & (main_2 > signal_2) & (main_2 <= oversold)
    buy_momentum = (main_0 > main_1 + 7) & (main_1 > main_2 + 7)
    stoch_buy = buy_oversold_1 | buy_oversold_2 | buy_momentum
    
    # STOCHASTIC SELL
    sell_overbought_1 = (main_2 > signal_2) & (main_1 < signal_1) & (main_1 >= overbought)
    sell_overbought_2 = (main_3 > signal_3) & (main_2 < signal_2) & (main_2 >= overbought)
    sell_momentum = (main_0 < main_1 - 7) & (main_1 < main_2 - 7)
    stoch_sell = sell_overbought_1 | sell_overbought_2 | sell_momentum
    
    # ADX
    adx_inst = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=adx_period)
    adx = adx_inst.adx()
    pdi = adx_inst.adx_pos()
    mdi = adx_inst.adx_neg()
    
    # Referencias ADX
    adx_0, adx_1, adx_2 = adx, adx.shift(1), adx.shift(2)
    pdi_0, pdi_1, pdi_2, pdi_3 = pdi, pdi.shift(1), pdi.shift(2), pdi.shift(3)
    mdi_0, mdi_1, mdi_2, mdi_3 = mdi, mdi.shift(1), mdi.shift(2), mdi.shift(3)
    
    # Pre-condición ADX
    adx_strong = (adx_0 > adx_limit) | (adx_1 > adx_limit) | (adx_1 - adx_2 > 5) | (adx_0 - adx_1 > 5)
    
    # ADX BUY
    buy_trend = (pdi_0 > pdi_2) & (pdi_1 > pdi_2) & (pdi_0 > pdi_1) & (mdi_0 < mdi_1) & (mdi_1 < mdi_2)
    buy_reversal = (mdi_2 < mdi_3) & (mdi_1 < mdi_2) & (mdi_0 < mdi_1) & (pdi_0 > pdi_2)
    adx_buy = adx_strong & (buy_trend | buy_reversal)
    
    # ADX SELL
    sell_trend = (mdi_0 > mdi_2) & (mdi_1 > mdi_2) & (mdi_0 > mdi_1) & (pdi_0 < pdi_1) & (pdi_1 < pdi_2)
    sell_reversal = (pdi_2 < pdi_3) & (pdi_1 < pdi_2) & (pdi_0 < pdi_1) & (mdi_0 > mdi_2)
    adx_sell = adx_strong & (sell_trend | sell_reversal)
    
    # Combinar señales
    buy_signals = stoch_buy & adx_buy
    sell_signals = stoch_sell & adx_sell
    
    return {
        'buy': buy_signals,
        'sell': sell_signals,
        'stoch_buy': stoch_buy,
        'stoch_sell': stoch_sell,
        'adx_buy': adx_buy,
        'adx_sell': adx_sell,
        'stoch_k': main,
        'stoch_d': signal,
        'adx': adx,
        'pdi': pdi,
        'mdi': mdi
    }


def main():
    parser = argparse.ArgumentParser(
        description='Comparar señales: Original vs SGRADT 5.0'
    )
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--output', type=str, default='./comparison')
    parser.add_argument('--head', type=int, default=100, 
                       help='Mostrar primeras N señales de cada tipo')
    
    args = parser.parse_args()
    csv_path = Path(args.csv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"COMPARACIÓN: Original vs SGRADT 5.0")
    print(f"{'='*80}")
    print(f"Archivo: {csv_path.name}\n")
    
    # Cargar datos
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else range(len(df))
    print(f"Datos cargados: {len(df)} barras\n")
    
    # Calcular señales con ambos métodos
    print("Calculando señales método ORIGINAL...")
    original = calculate_original_signals(df)
    
    print("Calculando señales método SGRADT 5.0...")
    sgradt = calculate_sgradt_signals(df)
    
    # Estadísticas
    print(f"\n{'='*80}")
    print(f"ESTADÍSTICAS DE SEÑALES")
    print(f"{'='*80}\n")
    
    orig_buy = original['buy'].sum()
    orig_sell = original['sell'].sum()
    sgradt_buy = sgradt['buy'].sum()
    sgradt_sell = sgradt['sell'].sum()
    
    print(f"{'Método':<20} {'BUY':>10} {'SELL':>10} {'Total':>10}")
    print(f"{'-'*52}")
    print(f"{'Original':<20} {orig_buy:>10} {orig_sell:>10} {orig_buy+orig_sell:>10}")
    print(f"{'SGRADT 5.0':<20} {sgradt_buy:>10} {sgradt_sell:>10} {sgradt_buy+sgradt_sell:>10}")
    print(f"{'-'*52}")
    print(f"{'Diferencia':<20} {sgradt_buy-orig_buy:>+10} {sgradt_sell-orig_sell:>+10} {(sgradt_buy+sgradt_sell)-(orig_buy+orig_sell):>+10}")
    
    # Análisis de coincidencia
    buy_both = (original['buy'] & sgradt['buy']).sum()
    buy_only_orig = (original['buy'] & ~sgradt['buy']).sum()
    buy_only_sgradt = (~original['buy'] & sgradt['buy']).sum()
    
    sell_both = (original['sell'] & sgradt['sell']).sum()
    sell_only_orig = (original['sell'] & ~sgradt['sell']).sum()
    sell_only_sgradt = (~original['sell'] & sgradt['sell']).sum()
    
    print(f"\n{'='*80}")
    print(f"ANÁLISIS DE COINCIDENCIA")
    print(f"{'='*80}\n")
    
    print(f"BUY Signals:")
    print(f"  Ambos métodos:      {buy_both:>6} ({buy_both/max(orig_buy,1)*100:>5.1f}% del original)")
    print(f"  Solo Original:      {buy_only_orig:>6}")
    print(f"  Solo SGRADT 5.0:    {buy_only_sgradt:>6}")
    
    print(f"\nSELL Signals:")
    print(f"  Ambos métodos:      {sell_both:>6} ({sell_both/max(orig_sell,1)*100:>5.1f}% del original)")
    print(f"  Solo Original:      {sell_only_orig:>6}")
    print(f"  Solo SGRADT 5.0:    {sell_only_sgradt:>6}")
    
    # Crear DataFrame de comparación
    comparison_df = pd.DataFrame({
        'timestamp': df['timestamp'],
        'close': df['close'],
        'orig_buy': original['buy'],
        'orig_sell': original['sell'],
        'sgradt_buy': sgradt['buy'],
        'sgradt_sell': sgradt['sell'],
        'sgradt_stoch_buy': sgradt['stoch_buy'],
        'sgradt_stoch_sell': sgradt['stoch_sell'],
        'sgradt_adx_buy': sgradt['adx_buy'],
        'sgradt_adx_sell': sgradt['adx_sell'],
        'orig_stoch_k': original['stoch_k'],
        'orig_stoch_d': original['stoch_d'],
        'sgradt_stoch_k': sgradt['stoch_k'],
        'sgradt_stoch_d': sgradt['stoch_d'],
        'orig_adx': original['adx'],
        'sgradt_adx': sgradt['adx'],
    })
    
    # Guardar comparación completa
    output_path = output_dir / f"{csv_path.stem}_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    print(f"\n✓ Comparación completa guardada: {output_path}")
    
    # Mostrar ejemplos de señales diferentes
    print(f"\n{'='*80}")
    print(f"EJEMPLOS DE SEÑALES DIFERENTES (primeras {args.head})")
    print(f"{'='*80}\n")
    
    # BUY solo en SGRADT
    buy_diff = comparison_df[comparison_df['sgradt_buy'] & ~comparison_df['orig_buy']].head(args.head)
    if len(buy_diff) > 0:
        print(f"BUY detectadas por SGRADT 5.0 pero NO por Original ({len(buy_diff)} de {buy_only_sgradt}):")
        print(buy_diff[['timestamp', 'close', 'sgradt_stoch_k', 'sgradt_stoch_d', 'sgradt_adx']].to_string(index=False))
    
    print("\n")
    
    # SELL solo en SGRADT
    sell_diff = comparison_df[comparison_df['sgradt_sell'] & ~comparison_df['orig_sell']].head(args.head)
    if len(sell_diff) > 0:
        print(f"SELL detectadas por SGRADT 5.0 pero NO por Original ({len(sell_diff)} de {sell_only_sgradt}):")
        print(sell_diff[['timestamp', 'close', 'sgradt_stoch_k', 'sgradt_stoch_d', 'sgradt_adx']].to_string(index=False))
    
    # Análisis de descomposición SGRADT
    print(f"\n{'='*80}")
    print(f"DESCOMPOSICIÓN SEÑALES SGRADT 5.0")
    print(f"{'='*80}\n")
    
    stoch_only_buy = (sgradt['stoch_buy'] & ~sgradt['adx_buy']).sum()
    adx_only_buy = (~sgradt['stoch_buy'] & sgradt['adx_buy']).sum()
    both_buy = (sgradt['stoch_buy'] & sgradt['adx_buy']).sum()
    
    stoch_only_sell = (sgradt['stoch_sell'] & ~sgradt['adx_sell']).sum()
    adx_only_sell = (~sgradt['stoch_sell'] & sgradt['adx_sell']).sum()
    both_sell = (sgradt['stoch_sell'] & sgradt['adx_sell']).sum()
    
    print(f"BUY Signals:")
    print(f"  Solo Stochastic:    {stoch_only_buy:>6} (no confirmado por ADX)")
    print(f"  Solo ADX:           {adx_only_buy:>6} (no confirmado por Stochastic)")
    print(f"  Ambos (final):      {both_buy:>6} ← señales válidas")
    
    print(f"\nSELL Signals:")
    print(f"  Solo Stochastic:    {stoch_only_sell:>6} (no confirmado por ADX)")
    print(f"  Solo ADX:           {adx_only_sell:>6} (no confirmado por Stochastic)")
    print(f"  Ambos (final):      {both_sell:>6} ← señales válidas")
    
    print(f"\n{'='*80}")
    print(f"CONCLUSIÓN")
    print(f"{'='*80}\n")
    
    if sgradt_buy + sgradt_sell < orig_buy + orig_sell:
        print("✓ SGRADT 5.0 es MÁS SELECTIVO (menos señales)")
        print("  → Filtra mejor las señales falsas")
        print("  → Mayor precisión esperada")
    elif sgradt_buy + sgradt_sell > orig_buy + orig_sell:
        print("✓ SGRADT 5.0 es MÁS SENSIBLE (más señales)")
        print("  → Detecta más oportunidades")
        print("  → Incluye condiciones de momentum")
    else:
        print("✓ Ambos métodos generan similar cantidad de señales")
    
    print(f"\nCoincidencia promedio: {(buy_both+sell_both)/(orig_buy+orig_sell)*100:.1f}%")
    print(f"Archivo de comparación: {output_path}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
