# CONFIGS

## EA_SGRADT70_ONNX

- EA_SGRADT70_ONNX.mq5
- train_sgradt70_strategy.py

### Preset for already working version

```
; ======== AI MODEL ========
InpModelName=ndx100_rates_m5_SGRADT70.onnx
InpMinConf=0.4
InpWindowSize=20
InpFeaturesPerBar=5
; ======== INFERENCE ========
InpInferSeconds=14
InpOneTradePerBar=true
InpReverseInfer=false
InpTrainParams=python src/python/train_sgradt70_strategy.py --window 20 --future 8 --min_profit_points 10 --adx_period 8 --adx_limit 24 --stoch_k 5 --stoch_oversold 30 --stoch_overbought 70 --n_iter 5 --csv csv/ndx100_rates_m5.csv
; ======== SESSION ========
InpStartHour=0
InpEndHour=24
; ======== EMA ========
InpEMAPeriod=9
InpUseEMAGate=true
; ======== STOCHASTIC ========
InpStochK=5
InpStochD=3
InpStochOversold=30.0
InpStochOverbought=70.0
; ======== ADX ========
InpADXPeriod=8
InpADXLimit=24.0
; ======== RISK ========
InpLot=0.01
InpMagic=70701
InpStopPoints=1000.0
InpTakePoints=1000.0
; ======== DISPLAY ========
InpShowPanel=true
```
