# Analizbot
Procesar datos
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class TradingAnalysisBot:
    def __init__(self):
        self.stock_data = {}
        self.crypto_data = {}
        self.models = {}
        self.predicciones = {}
        
    def recolectar_datos_externos(self, simbolos_acciones, simbolos_cripto, dias=365):
        """Recolecta datos externos de acciones y criptomonedas"""
        print("Recolectando datos externos...")
        
        # Recolectar datos de acciones
        for simbolo in simbolos_acciones:
            try:
                accion = yf.Ticker(simbolo)
                hist = accion.history(period=f"{dias}d")
                self.stock_data[simbolo] = hist
                print(f"Datos de {simbolo} recolectados correctamente")
            except Exception as e:
                print(f"Error obteniendo datos de {simbolo}: {str(e)}")
        
        # Recolectar datos de criptomonedas
        exchange = ccxt.binance()
        for simbolo in simbolos_cripto:
            try:
                since = exchange.parse8601((datetime.now() - timedelta(days=dias)).isoformat())
                ohlcv = exchange.fetch_ohlcv(simbolo, '1d', since=since, limit=dias)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                self.crypto_data[simbolo] = df
                print(f"Datos de {simbolo} recolectados correctamente")
            except Exception as e:
                print(f"Error obteniendo datos de {simbolo}: {str(e)}")
    
    def recolectar_datos_internos(self, archivo=None):
        """Recolecta datos internos desde archivos locales o bases de datos"""
        # Aquí puedes implementar la conexión a tu base de datos interna
        # o la carga de archivos locales con datos históricos
        print("Recolectando datos internos...")
        # Implementación pendiente según la fuente de datos interna
    
    def procesar_datos(self):
        """Procesa y limpia los datos recolectados"""
        print("Procesando datos...")
        
        # Procesar datos de acciones
        for simbolo, datos in self.stock_data.items():
            # Limpieza básica
            datos.dropna(inplace=True)
            # Calcular indicadores técnicos
            datos['MA20'] = datos['Close'].rolling(window=20).mean()
            datos['MA50'] = datos['Close'].rolling(window=50).mean()
            datos['RSI'] = self.calcular_rsi(datos['Close'])
            
        # Procesar datos de criptomonedas
        for simbolo, datos in self.crypto_data.items():
            datos.dropna(inplace=True)
            datos['MA20'] = datos['close'].rolling(window=20).mean()
            datos['MA50'] = datos['close'].rolling(window=50).mean()
            datos['RSI'] = self.calcular_rsi(datos['close'])
    
    def calcular_rsi(self, precios, periodo=14):
        """Calcula el índice de fuerza relativa (RSI)"""
        delta = precios.diff()
        ganancia = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
        perdida = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = ganancia / perdida
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def reconocer_patrones(self):
        """Reconoce patrones en los datos financieros"""
        print("Analizando patrones...")
        
        for tipo, mercados in [('Acciones', self.stock_data), ('Cripto', self.crypto_data)]:
            for simbolo, datos in mercados.items():
                print(f"\nAnalizando patrones en {simbolo} ({tipo}):")
                
                # Identificar tendencias
                tendencia = "Alcista" if datos['Close'].iloc[-1] > datos['Close'].iloc[-50] else "Bajista"
                print(f"Tendencia principal: {tendencia}")
                
                # Identificar cruces de medias móviles
                if datos['MA20'].iloc[-1] > datos['MA50'].iloc[-1] and datos['MA20'].iloc[-2] <= datos['MA50'].iloc[-2]:
                    print("Patrón: Cruce alcista de medias móviles (MA20 > MA50)")
                elif datos['MA20'].iloc[-1] < datos['MA50'].iloc[-1] and datos['MA20'].iloc[-2] >= datos['MA50'].iloc[-2]:
                    print("Patrón: Cruce bajista de medias móviles (MA20 < MA50)")
                
                # Analizar RSI
                rsi_actual = datos['RSI'].iloc[-1]
                if rsi_actual > 70:
                    print(f"RSI en zona de sobrecompra: {rsi_actual:.2f}")
                elif rsi_actual < 30:
                    print(f"RSI en zona de sobreventa: {rsi_actual:.2f}")
    
    def entrenar_modelos_prediccion(self):
        """Entrena modelos de machine learning para predicción"""
        print("\nEntrenando modelos de predicción...")
        
        for tipo, mercados in [('Acciones', self.stock_data), ('Cripto', self.crypto_data)]:
            for simbolo, datos in mercados.items():
                # Preparar datos para el modelo
                X = datos[['Open', 'High', 'Low', 'Volume', 'MA20', 'MA50', 'RSI']].values[:-1]
                y = datos['Close'].values[1:]
                
                # Eliminar filas con valores NaN
                valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
                X = X[valid_indices]
                y = y[valid_indices]
                
                if len(X) > 0 and len(y) > 0:
                    # Escalar datos
                    scaler_x = MinMaxScaler()
                    scaler_y = MinMaxScaler()
                    
                    X_scaled = scaler_x.fit_transform(X)
                    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
                    
                    # Dividir datos
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y_scaled, test_size=0.2, random_state=42
                    )
                    
                    # Entrenar modelo
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train.ravel())
                    
                    # Guardar modelo y scalers
                    self.models[simbolo] = {
                        'model': model,
                        'scaler_x': scaler_x,
                        'scaler_y': scaler_y
                    }
                    
                    score = model.score(X_test, y_test)
                    print(f"Modelo para {simbolo} entrenado. R²: {score:.4f}")
    
    def predecir(self, dias_futuro=7):
        """Realiza predicciones para los próximos días"""
        print(f"\nRealizando predicciones para los próximos {dias_futuro} días...")
        
        for tipo, mercados in [('Acciones', self.stock_data), ('Cripto', self.crypto_data)]:
            for simbolo, datos in mercados.items():
                if simbolo in self.models:
                    # Preparar últimos datos conocidos
                    ultimos_datos = datos[['Open', 'High', 'Low', 'Volume', 'MA20', 'MA50', 'RSI']].iloc[-1].values.reshape(1, -1)
                    
                    # Escalar
                    scaler_x = self.models[simbolo]['scaler_x']
                    scaler_y = self.models[simbolo]['scaler_y']
                    ultimos_datos_esc = scaler_x.transform(ultimos_datos)
                    
                    # Predecir
                    predicciones = []
                    current_data = ultimos_datos_esc
                    
                    for _ in range(dias_futuro):
                        pred_esc = self.models[simbolo]['model'].predict(current_data)
                        pred = scaler_y.inverse_transform(pred_esc.reshape(-1, 1))
                        predicciones.append(pred[0][0])
                        
                        # Actualizar datos para la siguiente predicción (simulación simplificada)
                        # En un sistema real, esto sería más complejo
                        current_data = np.roll(current_data, -1)
                        current_data[0, -1] = pred_esc[0]
                    
                    self.predicciones[simbolo] = predicciones
                    print(f"Predicción para {simbolo}: {predicciones}")
    
    def analizar_tendencias(self):
        """Analiza tendencias generales del mercado"""
        print("\nAnalizando tendencias del mercado...")
        
        # Aquí podrías integrar análisis de sentimiento, noticias, etc.
        # Por ahora, un análisis simple basado en los datos recolectados
        
        tendencia_acciones = []
        for simbolo, datos in self.stock_data.items():
            cambio = (datos['Close'].iloc[-1] - datos['Close'].iloc[-30]) / datos['Close'].iloc[-30] * 100
            tendencia_acciones.append(cambio)
        
        tendencia_cripto = []
        for simbolo, datos in self.crypto_data.items():
            cambio = (datos['close'].iloc[-1] - datos['close'].iloc[-30]) / datos['close'].iloc[-30] * 100
            tendencia_cripto.append(cambio)
        
        promedio_acciones = np.mean(tendencia_acciones)
        promedio_cripto = np.mean(tendencia_cripto)
        
        print(f"Tendencia promedio acciones (30 días): {promedio_acciones:.2f}%")
        print(f"Tendencia promedio cripto (30 días): {promedio_cripto:.2f}%")
        
        if promedio_acciones > 5:
            print("Mercado accionario con tendencia alcista fuerte")
        elif promedio_acciones > 0:
            print("Mercado accionario con tendencia alcista moderada")
        elif promedio_acciones > -5:
            print("Mercado accionario con tendencia bajista moderada")
        else:
            print("Mercado accionario con tendencia bajista fuerte")
    
    def generar_reporte(self):
        """Genera un reporte completo con los hallazgos"""
        print("\n" + "="*60)
        print("REPORTE COMPLETO DE ANÁLISIS FINANCIERO")
        print("="*60)
        
        # Incluir aquí un resumen de todas las análisis realizadas
        # patrones identificados, predicciones, tendencias, etc.
        
        print("Resumen de predicciones para los próximos 7 días:")
        for simbolo, preds in self.predicciones.items():
            cambio = (preds[-1] - preds[0]) / preds[0] * 100
            tendencia = "ALCISTA" if cambio > 0 else "BAJISTA"
            print(f"{simbolo}: {tendencia} ({cambio:.2f}%) - Valores: {[round(p, 2) for p in preds]}")
        
        print("\nRecomendaciones basadas en el análisis:")
        # Aquí podrías añadir recomendaciones de trading automáticas
        # basadas en los patrones y predicciones identificadas

# Ejemplo de uso
if __name__ == "__main__":
    bot = TradingAnalysisBot()
    
    # Definir símbolos a analizar
    acciones = ['AAPL', 'MSFT', 'TSLA']  # Ejemplo de acciones
    criptos = ['BTC/USDT', 'ETH/USDT']   # Ejemplo de criptomonedas
    
    # Recolectar datos
    bot.recolectar_datos_externos(acciones, criptos, dias=180)
    
    # Procesar datos
    bot.procesar_datos()
    
    # Analizar patrones
    bot.reconocer_patrones()
    
    # Analizar tendencias generales
    bot.analizar_tendencias()
    
    # Entrenar modelos de predicción
    bot.entrenar_modelos_prediccion()
    
    # Realizar predicciones
    bot.predecir(dias_futuro=7)
    
    # Generar reporte final
    bot.generar_reporte()
