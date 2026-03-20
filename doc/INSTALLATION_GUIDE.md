# SGRADT 6.0 - Guía de Instalación de Dependencias

---

## ✅ Dependencias Actualizadas (sin pandas_ta)

SGRADT 6.0 ahora usa **solo la librería `ta`**, sin necesidad de `pandas_ta`.

---

## 🚀 Instalación Rápida

### **Opción 1: Usando requirements.txt (RECOMENDADO)**

```bash
pip install -r requirements.txt
```

### **Opción 2: Instalación Manual**

```bash
pip install pandas numpy scikit-learn ta skl2onnx onnxruntime
```

---

## 📋 Lista de Dependencias

| Paquete | Versión Mínima | Propósito |
|---------|----------------|-----------|
| **pandas** | 1.3.0 | Manipulación de datos |
| **numpy** | 1.21.0 | Operaciones numéricas |
| **scikit-learn** | 1.0.0 | Random Forest y ML |
| **ta** | 0.10.0 | Indicadores técnicos |
| **skl2onnx** | 1.13.0 | Exportar modelo a ONNX |
| **onnxruntime** | 1.12.0 | Validación de ONNX |

---

## 🔍 Verificar Instalación

Ejecuta el script de validación:

```bash
python check_dependencies.py
```

**Salida esperada:**

```
======================================================================
SGRADT 6.0 - Validación de Dependencias
======================================================================

✅ pandas               - v1.5.3
✅ numpy                - v1.24.2
✅ scikit-learn         - v1.2.2
✅ ta                   - v0.10.2
✅ skl2onnx             - v1.14.0
✅ onnxruntime          - v1.14.1

======================================================================
✅ TODAS LAS DEPENDENCIAS ESTÁN INSTALADAS CORRECTAMENTE

Puedes ejecutar:
  python train_sgradt60_strategy.py --csv tus_datos.csv

======================================================================

Verificando componentes de 'ta'...

✅ ADXIndicator
✅ MACD
✅ StochasticOscillator
✅ RSIIndicator
✅ AverageTrueRange

✅ Todos los indicadores necesarios están disponibles

======================================================================
🎉 ¡TODO LISTO PARA USAR SGRADT 6.0!
======================================================================
```

---

## 🐛 Solución de Problemas

### **Error: "No module named 'ta'"**

```bash
pip install ta
```

Si ya está instalado pero da error:

```bash
pip install --upgrade ta
```

---

### **Error: "No module named 'pandas_ta'"**

**Solución:** Ignora este error. SGRADT 6.0 ya NO usa `pandas_ta`. 

Si aparece este error, asegúrate de estar usando la versión actualizada de `train_sgradt60_strategy.py`.

---

### **Error con MACD: "MACD() missing required argument"**

**Causa:** Versión antigua de `ta`

**Solución:**
```bash
pip install --upgrade ta
```

La librería `ta` debe ser versión 0.10.0 o superior.

---

### **Error: "Cannot import name 'MACD' from 'ta.trend'"**

**Causa:** Versión muy antigua de `ta`

**Solución:**
```bash
pip uninstall ta
pip install ta>=0.10.0
```

---

## 📦 Instalación en Entornos Virtuales

### **Usando venv (Python estándar)**

```bash
# Crear entorno virtual
python -m venv sgradt_env

# Activar (Linux/Mac)
source sgradt_env/bin/activate

# Activar (Windows)
sgradt_env\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### **Usando conda**

```bash
# Crear entorno
conda create -n sgradt python=3.9

# Activar
conda activate sgradt

# Instalar dependencias
pip install -r requirements.txt
```

---

## 🔬 Código de los Indicadores (referencia)

### **Cómo se usan en train_sgradt60_strategy.py**

```python
from ta.trend import ADXIndicator, MACD
from ta.momentum import StochasticOscillator, RSIIndicator
from ta.volatility import AverageTrueRange

# Stochastic
stoch = StochasticOscillator(
    high=df['high'], 
    low=df['low'], 
    close=df['close'],
    window=7,
    smooth_window=3
)
stoch_main = stoch.stoch()
stoch_signal = stoch.stoch_signal()

# ADX
adx_inst = ADXIndicator(
    high=df['high'], 
    low=df['low'], 
    close=df['close'], 
    window=8
)
adx = adx_inst.adx()
pdi = adx_inst.adx_pos()
mdi = adx_inst.adx_neg()

# RSI
rsi_inst = RSIIndicator(close=df['close'], window=14)
rsi = rsi_inst.rsi()

# MACD
macd_inst = MACD(
    close=df['close'],
    window_slow=26,
    window_fast=12,
    window_sign=9
)
macd_main = macd_inst.macd()
macd_signal = macd_inst.macd_signal()
macd_hist = macd_main - macd_signal

# ATR
atr_inst = AverageTrueRange(
    high=df['high'], 
    low=df['low'], 
    close=df['close'], 
    window=14
)
atr = atr_inst.average_true_range()
atr_pct = (atr / df['close']) * 100
```

---

## ✅ Ventajas de usar solo 'ta'

| Aspecto | pandas_ta | ta | ✅ Ventaja |
|---------|-----------|-----|-----------|
| **Instalación** | Compleja | Simple | Menos problemas |
| **Dependencias** | Muchas | Pocas | Más ligero |
| **Compatibilidad** | Variable | Estable | Menos errores |
| **Mantenimiento** | Irregular | Activo | Más confiable |
| **Documentación** | Limitada | Completa | Más fácil de usar |

---

## 📚 Documentación de 'ta'

Para más información sobre la librería `ta`:

- **GitHub**: https://github.com/bukosabino/ta
- **PyPI**: https://pypi.org/project/ta/
- **Docs**: https://technical-analysis-library-in-python.readthedocs.io/

---

## 🎯 Próximos Pasos

Una vez instaladas todas las dependencias:

1. **Verificar instalación:**
   ```bash
   python check_dependencies.py
   ```

2. **Entrenar el modelo:**
   ```bash
   python train_sgradt60_strategy.py --csv EUR_USD_H1.csv
   ```

3. **Copiar archivos a MT5**

4. **¡A operar!**

---

## 🆘 Soporte Adicional

Si sigues teniendo problemas:

1. Verifica tu versión de Python: `python --version` (mínimo 3.8)
2. Actualiza pip: `pip install --upgrade pip`
3. Instala una por una las dependencias para identificar cuál falla
4. Revisa los mensajes de error específicos

**Instalación paso a paso:**

```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install ta
pip install skl2onnx
pip install onnxruntime
```

Si alguna falla, el error te dirá exactamente qué pasa.

---

**Versión:** SGRADT 6.0 (sin pandas_ta)  
**Última actualización:** Marzo 2026  
**Status:** ✅ Probado y funcionando
