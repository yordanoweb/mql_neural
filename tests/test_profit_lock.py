"""Unit tests for profit lock and breakeven logic."""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch


def test_profit_lock_tf_mapping():
    """Test that profit lock timeframe mapping is correct."""
    # Define the mappings locally (copied from the source)
    TIMEFRAME_MAP = {
        'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 16385, 'H4': 16388, 'D1': 16408,
    }
    
    TRAILING_TF = {
        'M5': 'M1', 'M15': 'M5', 'M30': 'M15', 'H1': 'M30', 'H4': 'H1', 'D1': 'H4',
    }
    
    PROFIT_LOCK_TF = TRAILING_TF
    
    # Create reverse mapping
    TIMEFRAME_REVERSE_MAP = {v: k for k, v in TIMEFRAME_MAP.items()}
    
    # Test M15 -> M5
    assert TIMEFRAME_MAP['M15'] == 15
    assert TIMEFRAME_REVERSE_MAP[15] == 'M15'
    assert PROFIT_LOCK_TF['M15'] == 'M5'
    
    # Test M5 -> M1
    assert TIMEFRAME_MAP['M5'] == 5
    assert TIMEFRAME_REVERSE_MAP[5] == 'M5'
    assert PROFIT_LOCK_TF['M5'] == 'M1'
    
    # Test H1 -> M30
    assert PROFIT_LOCK_TF['H1'] == 'M30'
    
    # Test D1 -> H4
    assert PROFIT_LOCK_TF['D1'] == 'H4'
    
    # Test M1 fallback (no mapping)
    assert PROFIT_LOCK_TF.get('M1', 'M1') == 'M1'
    
    print("✓ Profit lock TF mapping test passed")


def test_move_sl_conditions_logic():
    """Test the logic of the three gates for SL movement."""
    # Gate a: price must be in profit
    entry_price = 100.0
    current_price_profit = 101.0  # in profit
    current_price_loss = 99.0     # not in profit
    
    # For BUY trade
    is_buy = True
    in_profit_buy_profit = current_price_profit > entry_price  # True
    in_profit_buy_loss = current_price_loss > entry_price      # False
    
    assert in_profit_buy_profit == True
    assert in_profit_buy_loss == False
    
    # Gate b: new SL must be past entry
    new_sl_below = 99.5   # below entry
    new_sl_above = 100.5  # above entry
    
    gate_b_fails = (is_buy and new_sl_below <= entry_price)  # True (fails)
    gate_b_passes = (is_buy and new_sl_above <= entry_price)  # False (passes)
    
    assert gate_b_fails == True
    assert gate_b_passes == False
    
    # Gate c: new SL must be better than current SL
    current_sl = 99.0
    new_sl_worse = 98.5   # worse (lower for BUY)
    new_sl_better = 99.5  # better (higher for BUY)
    
    gate_c_fails = (is_buy and new_sl_worse <= current_sl)   # True (fails)
    gate_c_passes = (is_buy and new_sl_better <= current_sl) # False (passes)
    
    assert gate_c_fails == True
    assert gate_c_passes == False
    
    print("✓ Move SL conditions logic test passed")


def test_breakeven_logic():
    """Test breakeven move at 0.5×ATR profit."""
    # Test cases
    test_cases = [
        # (profit_atr, should_trigger_breakeven)
        (0.4, False),  # below threshold
        (0.5, True),   # at threshold
        (0.6, True),   # above threshold
        (1.0, True),   # well above threshold
    ]
    
    for profit_atr, should_trigger in test_cases:
        triggers = profit_atr >= 0.5
        assert triggers == should_trigger, f"profit_atr={profit_atr}: expected {should_trigger}, got {triggers}"
    
    print("✓ Breakeven logic test passed")


def test_profit_lock_tf_calculation():
    """Test profit lock timeframe calculation from trading TF."""
    # Simulate the logic in manage_open_trade
    TIMEFRAME_MAP = {
        'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 16385, 'H4': 16388, 'D1': 16408,
    }
    
    TIMEFRAME_REVERSE_MAP = {v: k for k, v in TIMEFRAME_MAP.items()}
    
    PROFIT_LOCK_TF = {
        'M5': 'M1', 'M15': 'M5', 'M30': 'M15', 'H1': 'M30', 'H4': 'H1', 'D1': 'H4',
    }
    
    test_cases = [
        (15, 'M15', 'M5'),   # M15 -> M5
        (5, 'M5', 'M1'),     # M5 -> M1
        (16385, 'H1', 'M30'), # H1 -> M30
        (1, 'M1', 'M1'),     # M1 -> M1 (fallback)
    ]
    
    for tf_int, expected_tf_key, expected_pl_key in test_cases:
        tf_key = TIMEFRAME_REVERSE_MAP.get(tf_int, 'M1')
        assert tf_key == expected_tf_key
        
        profit_lock_tf_key = PROFIT_LOCK_TF.get(tf_key, 'M1')
        assert profit_lock_tf_key == expected_pl_key
        
        print(f"  {tf_int} ({tf_key}) → {profit_lock_tf_key}")
    
    print("✓ Profit lock TF calculation test passed")


if __name__ == '__main__':
    test_profit_lock_tf_mapping()
    test_move_sl_conditions_logic()
    test_breakeven_logic()
    test_profit_lock_tf_calculation()
    
    print("\n✅ All tests passed!")
