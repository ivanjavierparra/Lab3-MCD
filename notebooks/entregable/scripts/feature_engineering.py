import pandas as pd
import numpy as np
import os
from calendar import monthrange
import gc
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from scipy.stats import skew
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import calendar
from statsmodels.tsa.seasonal import seasonal_decompose
from neuralprophet import NeuralProphet
from tqdm import tqdm
from tslearn.metrics import cdist_dtw, dtw
from sklearn.cluster import KMeans

# from pmdarima.arima.utils import pacf

def calcular_cantidad_lags(desde, hasta):
    """
    Calcula la cantidad de lags necesarios entre dos fechas.
    """
    # Convertir fechas a formato datetime
    fdesde = pd.to_datetime(str(desde), format='%Y%m')
    fhasta = pd.to_datetime(str(hasta), format='%Y%m')
    
    # Calcular diferencia en meses
    lags = (fhasta.year - fdesde.year) * 12 + (fhasta.month - fdesde.month)
    return lags + 1  # +1 para incluir el mes actual



def get_lags(df, col, hasta=201912):
    """
    Calcula los lags de la columna indicada por 'col' hasta la fecha indicada.
    """
    df = df.sort_values(['product_id', 'periodo'])

    lags = calcular_cantidad_lags(201701, hasta)
    
    # Calcular lags
    for lag in range(1, lags):
        df[f'{col}_lag_{lag}'] = df.groupby('product_id')[col].shift(lag)
    
    # Liberar memoria
    gc.collect()
    
    return df


def get_lagsEspecificos(df, lags=[1, 2, 3, 6, 12, 24], col='tn'):
    """
    Calcula lags específicos de la columna indicada por 'col' hasta la fecha indicada.
    """
    df = df.sort_values(['product_id', 'periodo'])
    
    # Calcular lags específicos
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df.groupby('product_id')[col].shift(lag)
    
    # Liberar memoria
    gc.collect()
    
    return df


def get_delta_lags(df, col, hasta=201912):
    """
    Calcula los delta lags (diferencias entre lags consecutivos) para la columna 'col'.
    """
    
    # Ordenar
    df = df.sort_values(['product_id', 'periodo'])
    
    
    lags = calcular_cantidad_lags(201701, hasta)

    # Calcular delta lags
    for i in range(1, lags):
        lag_actual = i
        lag_anterior = i - 1
        if (lag_anterior == 0): continue
        df[f'{col}_delta_lag_{lag_actual}'] = df[f'{col}_lag_{lag_actual}'] - df[f'{col}_lag_{lag_anterior}']
    
    return df


def get_delta_lags_especificos(df, lags=[1, 2, 3, 6, 12, 24], col='tn'):
    """
    Calcula los delta lags (diferencias entre lags específicos) para la columna 'col'.
    """
    
    # Ordenar
    df = df.sort_values(['product_id', 'periodo'])
    
    # Calcular delta lags específicos
    for i in range(1, len(lags)):
        lag_actual = lags[i]
        lag_anterior = lags[i - 1]
        if (lag_anterior == 0): continue
        df[f'{col}_delta_lag_{lag_actual}'] = df[f'{col}_lag_{lag_actual}'] - df[f'{col}_lag_{lag_anterior}']
    
    return df


def get_rolling_means(df, col, hasta=201912):
    """
    Calcula medias móviles mensuales de la columna 'col' para las ventanas especificadas,
    usando solo datos hasta el período 'hasta'.
    """
    # Filtrar y ordenar
    df_historico = df[df['periodo'].astype(int) <= hasta].copy()
    df_historico = df_historico.sort_values(['product_id', 'periodo'])
    
    ventanas = calcular_cantidad_lags(201701, hasta)
    ventanas = ventanas + 1
    
    # Calcular medias móviles para cada ventana
    for ventana in range(1, ventanas):
        df_historico[f'{col}_rolling_mean_{ventana}'] = (
            df_historico.groupby('product_id')[col]
            .rolling(window=ventana, min_periods=ventana)
            .mean()
            .reset_index(level=0, drop=True)
        )
    
    # Combinar con el DataFrame original
    df = df.merge(
        df_historico[['product_id', 'periodo'] + [f'{col}_rolling_mean_{v}' for v in range(1, ventanas)]],
        on=['product_id', 'periodo'],
        how='left'
    )
    
    return df


def get_rolling_means_especificos(df, lags=[1, 2, 3, 6, 12, 24], col='tn'):
    """
    Calcula medias móviles mensuales de la columna 'col' para las ventanas especificadas,
    usando solo datos hasta el período 'hasta'.
    """
    # Filtrar y ordenar
    df_historico = df.copy()
    df_historico = df_historico.sort_values(['product_id', 'periodo'])
    
    # Calcular medias móviles para cada ventana específica
    for ventana in lags:
        df_historico[f'{col}_rolling_mean_{ventana}'] = (
            df_historico.groupby('product_id')[col]
            .rolling(window=ventana, min_periods=ventana)
            .mean()
            .reset_index(level=0, drop=True)
        )
    
    # Combinar con el DataFrame original
    df = df.merge(
        df_historico[['product_id', 'periodo'] + [f'{col}_rolling_mean_{v}' for v in lags]],
        on=['product_id', 'periodo'],
        how='left'
    )
    
    return df


def get_rolling_stds(df, col, hasta=201912):
    """
    Calcula desvíos estándar móviles mensuales de la columna 'col',
    usando solo datos hasta el período 'hasta'.
    """
    # Filtrar y ordenar
    df_historico = df[df['periodo'].astype(int) <= hasta].copy()
    df_historico = df_historico.sort_values(['product_id', 'periodo'])
    
    ventanas = calcular_cantidad_lags(201701, hasta)
    ventanas = ventanas + 1
    
    # Calcular desvíos estándar móviles para cada ventana
    for ventana in range(1, ventanas):
        df_historico[f'{col}_rolling_std_{ventana}'] = (
            df_historico.groupby('product_id')[col]
            .rolling(window=ventana, min_periods=ventana)  # min_periods=ventana para evitar valores con datos insuficientes
            .std(ddof=0)  # ddof=0 para desviación estándar poblacional
            .reset_index(level=0, drop=True)
        )
    
    # Combinar con el DataFrame original
    df = df.merge(
        df_historico[['product_id', 'periodo'] + [f'{col}_rolling_std_{v}' for v in range(1, ventanas)]],
        on=['product_id', 'periodo'],
        how='left'
    )
    
    return df


def get_rolling_stds_especificos(df, lags=[1, 2, 3, 6, 12, 24], col='tn'):
    """
    Calcula desvíos estándar móviles mensuales de la columna 'col' para las ventanas especificadas,
    usando solo datos hasta el período 'hasta'.
    """
    # Filtrar y ordenar
    df_historico = df.copy()
    df_historico = df_historico.sort_values(['product_id', 'periodo'])
    
    # Calcular desvíos estándar móviles para cada ventana específica
    for ventana in lags:
        df_historico[f'{col}_rolling_std_{ventana}'] = (
            df_historico.groupby('product_id')[col]
            .rolling(window=ventana, min_periods=ventana)  # min_periods=ventana para evitar valores con datos insuficientes
            .std(ddof=0)  # ddof=0 para desviación estándar poblacional
            .reset_index(level=0, drop=True)
        )
    
    # Combinar con el DataFrame original
    df = df.merge(
        df_historico[['product_id', 'periodo'] + [f'{col}_rolling_std_{v}' for v in lags]],
        on=['product_id', 'periodo'],
        how='left'
    )
    
    return df


def get_rolling_medians(df, col, hasta=201912):
    """
    Calcula medianas móviles mensuales de la columna 'col' para las ventanas especificadas,
    usando solo datos hasta el período 'hasta'.
    """
    # Filtrar y ordenar
    df_historico = df[df['periodo'].astype(int) <= hasta].copy()
    df_historico = df_historico.sort_values(['product_id', 'periodo'])
    
    ventanas = calcular_cantidad_lags(201701, hasta)
    ventanas = ventanas + 1
    
    # Calcular medianas móviles para cada ventana
    for ventana in range(1, ventanas):
        df_historico[f'{col}_rolling_median_{ventana}'] = (
            df_historico.groupby('product_id')[col]
            .rolling(window=ventana, min_periods=ventana)
            .median()  # Cambio clave: usando median() en lugar de mean()
            .reset_index(level=0, drop=True)
        )
    
    # Combinar con el DataFrame original
    df = df.merge(
        df_historico[['product_id', 'periodo'] + [f'{col}_rolling_median_{v}' for v in range(1, ventanas)]],
        on=['product_id', 'periodo'],
        how='left'
    )
    
    return df


def get_rolling_medians_especificos(df, lags=[1, 2, 3, 6, 12, 24], col='tn'):
    """
    Calcula medianas móviles mensuales de la columna 'col' para las ventanas especificadas,
    usando solo datos hasta el período 'hasta'.
    """
    # Filtrar y ordenar
    df_historico = df.copy()
    df_historico = df_historico.sort_values(['product_id', 'periodo'])
    
    # Calcular medianas móviles para cada ventana específica
    for ventana in lags:
        df_historico[f'{col}_rolling_median_{ventana}'] = (
            df_historico.groupby('product_id')[col]
            .rolling(window=ventana, min_periods=ventana)
            .median()  # Cambio clave: usando median() en lugar de mean()
            .reset_index(level=0, drop=True)
        )
    
    # Combinar con el DataFrame original
    df = df.merge(
        df_historico[['product_id', 'periodo'] + [f'{col}_rolling_median_{v}' for v in lags]],
        on=['product_id', 'periodo'],
        how='left'
    )
    
    return df


def get_rolling_mins(df, col, hasta=201912):
    """
    Calcula mínimos móviles mensuales de la columna 'col',
    usando solo datos hasta el período 'hasta'.
    """
    # Filtrar y ordenar
    df_historico = df[df['periodo'].astype(int) <= hasta].copy()
    df_historico = df_historico.sort_values(['product_id', 'periodo'])
    
    ventanas = calcular_cantidad_lags(201701, hasta)
    ventanas = ventanas + 1
    
    # Calcular mínimos móviles para cada ventana
    for ventana in range(1, ventanas):
        df_historico[f'{col}_rolling_min_{ventana}'] = (
            df_historico.groupby('product_id')[col]
            .rolling(window=ventana, min_periods=ventana)  # min_periods=ventana para NaN con datos insuficientes
            .min()
            .reset_index(level=0, drop=True)
        )
    
    # Combinar con el DataFrame original
    df = df.merge(
        df_historico[['product_id', 'periodo'] + [f'{col}_rolling_min_{v}' for v in range(1, ventanas)]],
        on=['product_id', 'periodo'],
        how='left'
    )
    
    
    return df

def get_rolling_mins_especificos(df, lags=[1, 2, 3, 6, 12, 24], col='tn'):  
    """
    Calcula mínimos móviles mensuales de la columna 'col' para las ventanas especificadas,
    usando solo datos hasta el período 'hasta'.
    """
    # Filtrar y ordenar
    df_historico = df.copy()
    df_historico = df_historico.sort_values(['product_id', 'periodo'])
    
    # Calcular mínimos móviles para cada ventana específica
    for ventana in lags:
        df_historico[f'{col}_rolling_min_{ventana}'] = (
            df_historico.groupby('product_id')[col]
            .rolling(window=ventana, min_periods=ventana)  # min_periods=ventana para NaN con datos insuficientes
            .min()
            .reset_index(level=0, drop=True)
        )
    
    # Combinar con el DataFrame original
    df = df.merge(
        df_historico[['product_id', 'periodo'] + [f'{col}_rolling_min_{v}' for v in lags]],
        on=['product_id', 'periodo'],
        how='left'
    )
    
    return df

def get_rolling_maxs(df, col, hasta=201912):
    """
    Calcula máximos móviles mensuales de la columna 'col',
    usando solo datos hasta el período 'hasta'.
    """
    # Filtrar y ordenar
    df_historico = df[df['periodo'].astype(int) <= hasta].copy()
    df_historico = df_historico.sort_values(['product_id', 'periodo'])
    
    ventanas = calcular_cantidad_lags(201701, hasta)
    ventanas = ventanas + 1
    
    # Calcular mínimos móviles para cada ventana
    for ventana in range(1, ventanas):
        df_historico[f'{col}_rolling_max_{ventana}'] = (
            df_historico.groupby('product_id')[col]
            .rolling(window=ventana, min_periods=ventana)  # min_periods=ventana para NaN con datos insuficientes
            .max()
            .reset_index(level=0, drop=True)
        )
    
    # Combinar con el DataFrame original
    df = df.merge(
        df_historico[['product_id', 'periodo'] + [f'{col}_rolling_max_{v}' for v in range(1, ventanas)]],
        on=['product_id', 'periodo'],
        how='left'
    )
    
    
    return df


def get_rolling_maxs_especificos(df, lags=[1, 2, 3, 6, 12, 24], col='tn'):
    """
    Calcula máximos móviles mensuales de la columna 'col' para las ventanas especificadas,
    usando solo datos hasta el período 'hasta'.
    """
    # Filtrar y ordenar
    df_historico = df.copy()
    df_historico = df_historico.sort_values(['product_id', 'periodo'])
    
    # Calcular máximos móviles para cada ventana específica
    for ventana in lags:
        df_historico[f'{col}_rolling_max_{ventana}'] = (
            df_historico.groupby('product_id')[col]
            .rolling(window=ventana, min_periods=ventana)  # min_periods=ventana para NaN con datos insuficientes
            .max()
            .reset_index(level=0, drop=True)
        )
    
    # Combinar con el DataFrame original
    df = df.merge(
        df_historico[['product_id', 'periodo'] + [f'{col}_rolling_max_{v}' for v in lags]],
        on=['product_id', 'periodo'],
        how='left'
    )
    
    return df


def get_rolling_skewness(df, col, hasta=201912):
    """
    Calcula el skewness móvil mensual de la columna 'col' para ventanas especificadas,
    usando solo datos hasta el período 'hasta'.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas 'product_id', 'periodo' y 'col'.
        col (str): Nombre de la columna a analizar.
        hasta (int): Fecha límite en formato YYYYMM (ej: 201912 para diciembre 2019).
    
    Returns:
        pd.DataFrame: DataFrame original con columnas añadidas de skewness móvil.
    """
    # Filtrar y ordenar datos históricos
    df_historico = df[df['periodo'].astype(int) <= hasta].copy()
    df_historico = df_historico.sort_values(['product_id', 'periodo'])
    
    # Calcular número de ventanas posibles
    ventanas = calcular_cantidad_lags(201701, hasta) + 1
    
    # Calcular skewness móvil para cada ventana
    for ventana in range(1, ventanas):
        df_historico[f'{col}_rolling_skew_{ventana}'] = (
            df_historico.groupby('product_id')[col]
            .rolling(window=ventana, min_periods=ventana)
            .apply(skew, raw=True)  # Usamos scipy.stats.skew
            .reset_index(level=0, drop=True)
        )
    
    # Combinar con el DataFrame original
    df = df.merge(
        df_historico[['product_id', 'periodo'] + [f'{col}_rolling_skew_{v}' for v in range(1, ventanas)]],
        on=['product_id', 'periodo'],
        how='left'
    )
    
    return df


def get_rolling_skewness_especificos(df, lags=[1, 2, 3, 6, 12, 24], col='tn'):
    """
    Calcula el skewness móvil mensual de la columna 'col' para ventanas específicas,
    usando solo datos hasta el período 'hasta'.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas 'product_id', 'periodo' y 'col'.
        lags (list): Lista de ventanas específicas para calcular skewness.
        col (str): Nombre de la columna a analizar.
    
    Returns:
        pd.DataFrame: DataFrame original con columnas añadidas de skewness móvil.
    """
    # Filtrar y ordenar datos históricos
    df_historico = df.copy()
    df_historico = df_historico.sort_values(['product_id', 'periodo'])
    
    # Calcular skewness móvil para cada ventana específica
    for ventana in lags:
        df_historico[f'{col}_rolling_skew_{ventana}'] = (
            df_historico.groupby('product_id')[col]
            .rolling(window=ventana, min_periods=ventana)
            .apply(skew, raw=True)  # Usamos scipy.stats.skew
            .reset_index(level=0, drop=True)
        )
    
    # Combinar con el DataFrame original
    df = df.merge(
        df_historico[['product_id', 'periodo'] + [f'{col}_rolling_skew_{v}' for v in lags]],
        on=['product_id', 'periodo'],
        how='left'
    )
    
    return df


def get_autocorrelaciones(df, col, hasta=201912):
    """
    Calcula autocorrelaciones para cada combinación de producto y periodo,
    considerando todos los lags posibles hasta la fecha límite.
    """
    # Ordenar por producto y periodo
    df = df.sort_values(['product_id', 'periodo'])
    
    # Lista para almacenar resultados
    resultados = []
    
    for (producto, periodo), grupo in df.groupby(['product_id', 'periodo']):
        # Filtrar datos históricos hasta el periodo actual (inclusive)
        datos_historicos = df[
            (df['product_id'] == producto) & 
            (df['periodo'] <= periodo)
        ][col].dropna()
        
        # Calcular número máximo de lags posibles (periodos disponibles - 1)
        max_lags_posibles = len(datos_historicos) - 1
        max_lags_posibles = max(max_lags_posibles, 0)  # Asegurar no negativo
        
        # Diccionario para este registro
        registro = {
            'product_id': producto,
            'periodo': periodo
        }
        
        # Calcular autocorrelación para cada lag posible
        for lag in range(1, max_lags_posibles + 1):
            autocorr = datos_historicos.autocorr(lag=lag)
            registro[f'autocorr_lag_{lag}'] = autocorr
        
        resultados.append(registro)
    
    # Convertir a DataFrame y rellenar NaNs para lags no disponibles
    df_resultado = pd.DataFrame(resultados)
    df = df.merge(df_resultado, on=['product_id', 'periodo'], how='left')
    
    
    return df







def generar_ids(df):
    """
    Genera un ID único para cada fila del DataFrame basado en 'product_id' y 'periodo'.
    """
    df = df.sort_values(['periodo', 'product_id'])
    df['id'] = df.groupby(['product_id']).cumcount() + 1
    return df
    


def get_componentesTemporales(df):
    """
    Extrae componentes temporales de la columna 'periodo' del DataFrame.
    """
    df['periodo_dt'] = pd.to_datetime(df['periodo'], format='%Y%m')
    
    # Extraer año y mes
    df['year'] = df['periodo_dt'].dt.year
    df['month'] = df['periodo_dt'].dt.month
    df['quarter'] = df['periodo_dt'].dt.quarter
    df['semester'] = np.where(df['month'] <= 6, 1, 2)
    
    # Días en el mes
    df['dias_en_mes'] = df['periodo_dt'].apply(lambda x: monthrange(x.year, x.month)[1])
    
    # Extraer el número de semana del año
    df['week_of_year'] = df['periodo_dt'].dt.isocalendar().week
    
    
    # Efectos de fin de año
    df['year_end'] = np.where(df['month'].isin([11, 12]), 1, 0)
    df['year_start'] = np.where(df['month'].isin([1, 2]), 1, 0)
    
    
    # Indicadores estacionales
    df['season'] = df['month'] % 12 // 3 + 1  # 1:Invierno, 2:Primavera, etc.
    
        
    # Componentes cíclicos (para continuidad temporal) (para capturar patrones estacionales)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter']/4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter']/4)
    
    
    # Opcional: Día del año (para estacionalidad)
    df['dia_del_year'] = df['periodo_dt'].dt.dayofyear
    df['dia_del_year_sin'] = np.sin(2 * np.pi * df['dia_del_year']/365)
    df['dia_del_year_cos'] = np.cos(2 * np.pi * df['dia_del_year']/365)
        
    
    df.drop(columns=['periodo_dt'], inplace=True)  # Eliminar la columna temporal
    
    return df





def get_prophet_features(df, target_col):
    """
    Crea features de Prophet.
    target_col: Columna objetivo para el modelo Prophet.
    Esta función utiliza Prophet para extraer componentes aditivos y multiplicativos de la serie temporal
    y los combina con el DataFrame original.
    """
    
    ruta_archivo = f"./features_prophet_completo_{target_col}.csv"
    
    if os.path.exists(ruta_archivo) and ruta_archivo.endswith('.csv'):        
        features_final = pd.read_csv(ruta_archivo, sep=',', encoding='utf-8')
        features_final['ds'] = pd.to_datetime(features_final['ds']).dt.strftime('%Y%m').astype(int)
        features_final.drop(columns=['y','yhat1','yhat2'], inplace=True)
        df = df.merge(features_final, on=['periodo', 'product_id'], how='left')
        return df
    
    
    date_col = 'periodo'
    product_col = 'product_id'
    


    df['ds'] = pd.to_datetime(df[date_col].astype(str), format='%Y%m')
    df = df.sort_values([product_col, 'ds'])

    all_features = []

    for product in df[product_col].unique():
        product_df = df[df[product_col] == product].copy()
        
        # --- Modelo ADITIVO ---
        model_add = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive'
        )
        model_add.fit(product_df[['ds', target_col]].rename(columns={target_col: 'y'}))
        forecast_add = model_add.make_future_dataframe(periods=0)
        components_add = model_add.predict(forecast_add)
        
        # --- Modelo MULTIPLICATIVO ---
        model_mult = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        model_mult.fit(product_df[['ds', target_col]].rename(columns={target_col: 'y'}))
        forecast_mult = model_mult.make_future_dataframe(periods=0)
        components_mult = model_mult.predict(forecast_mult)
        
        # --- Combinar componentes ---
        features = pd.DataFrame({'ds': components_add['ds']})
        
        # Extraer componentes ADITIVOS
        features['trend_add'] = components_add['trend']
        features['yearly_add'] = components_add['yearly']
        features['additive_terms'] = components_add['additive_terms']  # Suma de todos los términos aditivos
        
        # Extraer componentes MULTIPLICATIVOS (solo existen en mode multiplicativo)
        features['trend_mult'] = components_mult['trend']
        features['yearly_mult'] = components_mult['yearly']
        features['multiplicative_terms'] = components_mult['multiplicative_terms']  # Producto de términos multiplicativos
        
        features['product_id'] = product
        
        # Escalar features
        scaler = StandardScaler()
        feature_cols = ['trend_add', 'yearly_add', 'additive_terms', 
                        'trend_mult', 'yearly_mult', 'multiplicative_terms']
        features[feature_cols] = scaler.fit_transform(features[feature_cols])
        
        all_features.append(features)

    # Combinar todos los productos
    prophet_features = pd.concat(all_features)
    
    

    # Unir con los datos originales
    df = df.merge(prophet_features, on=['ds', 'product_id'])
    
    return df




def get_prophet_features_v2(df, target_col):
    """
    Crea features de Prophet de manera robusta.
    
    Args:
        df: DataFrame con columnas 'periodo' (int YYYYMM) y 'product_id'
        target_col: Nombre de la columna objetivo
        
    Returns:
        DataFrame con features de Prophet añadidas
    """
    # Verificación de instalación correcta
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Por favor instala las dependencias correctas:\n"
                        "pip uninstall pystan prophet fbprophet\n"
                        "pip install prophet")
    
    # Cache de features
    ruta_archivo = f"./features_prophet_completo_{target_col}.csv"
    if os.path.exists(ruta_archivo):
        try:
            features_final = pd.read_csv(ruta_archivo, parse_dates=['ds'])
            features_final['periodo'] = features_final['ds'].dt.strftime('%Y%m').astype(int)
            features_final.drop(columns=['ds', 'y', 'yhat1', 'yhat2'], inplace=True, errors='ignore')
            return df.merge(features_final, on=['periodo', 'product_id'], how='left')
        except Exception as e:
            print(f"Error cargando cache: {e}")

    # Preparación de datos
    df = df.copy()
    df['ds'] = pd.to_datetime(df['periodo'].astype(str), format='%Y%m')
    df = df.sort_values(['product_id', 'ds'])
    
    all_features = []
    feature_cols = [
        'trend_add', 'yearly_add', 'additive_terms',
        'trend_mult', 'yearly_mult', 'multiplicative_terms'
    ]

    for product in df['product_id'].unique():
        try:
            product_df = df[df['product_id'] == product].dropna(subset=[target_col])
            if len(product_df) < 2:  # Mínimo 2 puntos para Prophet
                continue

            # Modelo Aditivo
            model_add = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='additive'
            )
            model_add.fit(product_df[['ds', target_col]].rename(columns={target_col: 'y'}))
            
            # Modelo Multiplicativo
            model_mult = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            model_mult.fit(product_df[['ds', target_col]].rename(columns={target_col: 'y'}))

            # Predicciones
            future = model_add.make_future_dataframe(periods=0)
            components_add = model_add.predict(future)
            components_mult = model_mult.predict(future)

            # Construcción de features
            features = pd.DataFrame({
                'product_id': product,
                'ds': components_add['ds'],
                'trend_add': components_add['trend'],
                'yearly_add': components_add['yearly'],
                'additive_terms': components_add['additive_terms'],
                'trend_mult': components_mult['trend'],
                'yearly_mult': components_mult['yearly'],
                'multiplicative_terms': components_mult['multiplicative_terms']
            })

            # Escalado robusto
            scaler = StandardScaler()
            features[feature_cols] = scaler.fit_transform(features[feature_cols])
            all_features.append(features)

        except Exception as e:
            print(f"Error procesando producto {product}: {str(e)}")
            continue

    if not all_features:
        raise ValueError("No se pudo generar features para ningún producto")

    # Consolidación
    prophet_features = pd.concat(all_features)
    prophet_features['periodo'] = prophet_features['ds'].dt.strftime('%Y%m').astype(int)
    
    # Guardar cache
    try:
        prophet_features.to_csv(ruta_archivo, index=False)
    except Exception as e:
        print(f"Error guardando cache: {e}")

    return df.merge(
        prophet_features.drop(columns=['ds']),
        on=['periodo', 'product_id'],
        how='left'
    )





def get_prophet_features_sin_data_leakage(df, target_col, train_mask):
    """
    Crea features de Prophet y las escala SIN LEAKAGE.
    
    Parámetros:
    - df: DataFrame con los datos.
    - target_col: Columna objetivo para Prophet.
    - train_mask: Máscara booleana que indica qué filas son de entrenamiento.
                  Ejemplo: train_mask = (df['ds'] <= '2019-09-01')
    """
    date_col = 'periodo'
    product_col = 'product_id'
    
    df['ds'] = pd.to_datetime(df[date_col].astype(str), format='%Y%m')
    df = df.sort_values([product_col, 'ds'])
    
    all_features = []
    
    for product in df[product_col].unique():
        product_df = df[df[product_col] == product].copy()
        
        # --- Modelo ADITIVO ---
        model_add = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive'
        )
        model_add.fit(product_df[['ds', target_col]].rename(columns={target_col: 'y'}))
        forecast_add = model_add.make_future_dataframe(periods=0)
        components_add = model_add.predict(forecast_add)
        
        # --- Modelo MULTIPLICATIVO ---
        model_mult = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        model_mult.fit(product_df[['ds', target_col]].rename(columns={target_col: 'y'}))
        forecast_mult = model_mult.make_future_dataframe(periods=0)
        components_mult = model_mult.predict(forecast_mult)
        
        # --- Combinar componentes ---
        features = pd.DataFrame({'ds': components_add['ds']})
        features['trend_add'] = components_add['trend']
        features['yearly_add'] = components_add['yearly']
        features['additive_terms'] = components_add['additive_terms']
        features['trend_mult'] = components_mult['trend']
        features['yearly_mult'] = components_mult['yearly']
        features['multiplicative_terms'] = components_mult['multiplicative_terms']
        features['product_id'] = product
        
        # --- Escalar SIN LEAKAGE ---
        # 1. Separar features de train y test para este producto
        product_train_mask = (df[product_col] == product) & train_mask
        train_features = features[features['ds'].isin(df.loc[product_train_mask, 'ds'])]
        test_features = features[~features['ds'].isin(df.loc[product_train_mask, 'ds'])]
        
        # 2. Escalar SOLO con datos de train
        scaler = StandardScaler()
        feature_cols = ['trend_add', 'yearly_add', 'additive_terms', 
                       'trend_mult', 'yearly_mult', 'multiplicative_terms']
        
        if not train_features.empty:
            train_features[feature_cols] = scaler.fit_transform(train_features[feature_cols])
            if not test_features.empty:
                test_features[feature_cols] = scaler.transform(test_features[feature_cols])
        
        # 3. Combinar train y test escalados
        features_scaled = pd.concat([train_features, test_features])
        all_features.append(features_scaled)
    
    prophet_features = pd.concat(all_features)
    df = df.merge(prophet_features, on=['ds', 'product_id'])
    return df



def get_anomaliasPoliticas(df):
    """
    Crea variables de anomalías políticas basadas en fechas clave.
    """
    # Definir fechas clave
    fechas_clave = {
        'elecciones_legislativas_1': 201708, #  El oficialismo (Cambiemos) pierde en Buenos Aires frente a Cristina Kirchner (Unidad Ciudadana), quien se convierte en senadora.
        'elecciones_legislativas_2': 201710,
        'crisis_cambiaria_1': 201806,
        'crisis_cambiaria_2': 201808, #  El peso argentino se devalúa un 20% frente al dólar en un solo día, marcando el inicio de una crisis cambiaria.
        'las_paso_2019_1': 201906, 
        'las_paso_2019_2': 201908, # Las PASO de 2019 marcan un cambio significativo en el panorama político, con la victoria de la oposición (Frente de Todos) sobre el oficialismo (Cambiemos).
        'devaluacion_post_PASO': 201909, # La crisis económica se agudiza con una inflación que supera el 50% anual y una recesión prolongada.
        'elecciones_presidenciales_2019': 201911, # Elecciones presidenciales donde Alberto Fernández (Frente de Todos) derrota a Mauricio Macri (Cambiemos).        
    }
    
    # Crear columnas de anomalías
    for key, fecha in fechas_clave.items():
        df[key] = np.where(df['periodo'] == fecha, 1, 0)
    
    return df


def get_IPC(df):
    """
    Agrega el IPC (Índice de Precios al Consumidor) al DataFrame.
    """
    
    ipc =  pd.read_csv("./datasets/ipc.csv", sep=';', encoding='utf-8')
    df = df.merge(ipc, on='periodo', how='left')
    
    return df


def get_dolar(df):
    """
    Agrega el valor del dólar oficial al DataFrame.
    """
    
    dolar =  pd.read_csv("./datasets/dolar_oficial_bna.csv", sep=';', encoding='utf-8')
    
    df = df.merge(dolar, on='periodo', how='left')
    
    return df


def get_mes_receso_escolar(df):
    """ 
    Crea una columna que indica si el mes es de receso escolar.
    """
    
    df['periodo_dt'] = pd.to_datetime(df['periodo'], format='%Y%m')
    df['anio_1'] = df['periodo_dt'].dt.year
    df['mes_1'] = df['periodo_dt'].dt.month
    
    # Definir meses de receso por año (sin días específicos)
    receso_por_mes = {
        # Vacaciones de invierno: julio (siempre)
        'meses_invierno': [7],  
        # Vacaciones de verano: enero, febrero y diciembre
        'meses_verano': [1, 2, 12],  
        # Semana Santa: abril o marzo (depende del año)
        'semana_santa': {
            2017: [4],  # Abril 2017
            2018: [3],  # Marzo 2018
            2019: [4],   # Abril 2019
        }
    }
    
    # Función para aplicar a cada fila
    def es_receso(row):
        if row['mes_1'] in receso_por_mes['meses_invierno']:
            return 1
        elif row['mes_1'] in receso_por_mes['meses_verano']:
            return 1
        elif row['anio_1'] in receso_por_mes['semana_santa']:
            if row['mes_1'] in receso_por_mes['semana_santa'][row['anio_1']]:
                return 1
        return 0
    
    # Aplicar la función y crear la columna
    df['receso_escolar'] = df.apply(es_receso, axis=1)

    df.drop(columns=['periodo_dt', 'anio_1', 'mes_1'], inplace=True)  # Eliminar la columna temporal
    
    del receso_por_mes
    gc.collect()
    
    return df


def mes_con_feriado(df):
    
    # Convertir a datetime y extraer año y mes
    df['periodo_dt'] = pd.to_datetime(df['periodo'], format='%Y%m')
    df['anio_1'] = df['periodo_dt'].dt.year
    df['mes_1'] = df['periodo_dt'].dt.month
    
    # Definir feriados nacionales (día, mes, año opcional)
    feriados = [
        # Feriados fijos (día, mes)
        (1, 1),    # Año Nuevo
        (24, 3),   # Día Nacional de la Memoria
        (2, 4),    # Día del Veterano y Caídos en Malvinas (hasta 2019: 2 de abril)
        (1, 5),    # Día del Trabajador
        (25, 5),   # Revolución de Mayo
        (20, 6),   # Paso a la Inmortalidad de Belgrano
        (9, 7),    # Día de la Independencia
        (17, 8),   # Paso a la Inmortalidad de San Martín
        (12, 10),  # Día del Respeto a la Diversidad Cultural
        (20, 11),  # Día de la Soberanía Nacional
        (8, 12),   # Inmaculada Concepción
        (25, 12),  # Navidad
        
        # Feriados móviles (Semana Santa: jueves y viernes Santo)
        # (2017: 13-14/4; 2018: 29-30/3; 2019: 18-19/4)
    ]

    # Función para verificar si un mes contiene feriados
    def tiene_feriado(row):
        anio = row['anio_1']
        mes = row['mes_1']
        
        # Feriados fijos (sin año específico)
        for dia, mes_feriado in feriados:
            if mes == mes_feriado:
                return 1
        
        # Feriados móviles (Semana Santa por año)
        if anio == 2017 and mes == 4:  # Abril 2017 (13-16/4)
            return 1
        elif anio == 2018 and mes == 3:  # Marzo 2018 (29/3 - 1/4)
            return 1
        elif anio == 2019 and mes == 4:  # Abril 2019 (18-21/4)
            return 1
        
        return 0

    # Aplicar la función y crear columna
    df['contiene_feriado'] = df.apply(tiene_feriado, axis=1)
    
    return df
    
    
    
def get_clustersDTW(df, n_clusters=50):
    """
    Agrupa series temporales de productos utilizando DTW (Dynamic Time Warping).
    """
    
    # 1. Preprocesar: pivotear la tabla para tener 1 fila = 1 producto
    df['periodo_dt'] = pd.to_datetime(df['periodo'], format='%Y%m')
    df = df.sort_values(['product_id', 'periodo'])

    # Asegurar consistencia temporal
    pivot_df = df.pivot(index='product_id', columns='periodo_dt', values='tn').fillna(0)

    # 2. Escalar las series (opcional pero recomendado)
    X = pivot_df.values
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)

    # 3. Clustering con DTW
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=0)
    labels = model.fit_predict(X_scaled)

    # 4. Añadir etiquetas al DataFrame original
    pivot_df['cluster'] = labels
    
    
    df.drop(columns=['periodo_dt'], inplace=True) 
    
    # 5. Unir los clusters al DataFrame original
    df = df.merge(pivot_df[['cluster']], on='product_id', how='left')
    
    return df






def get_neural_prophet_features(df_entrada):
   
    """
    Extrae características utilizando NeuralProphet para series temporales de productos.
    """
    
    ruta_archivo = "./features_neuralprophet_completo.csv"
    
    if os.path.exists(ruta_archivo) and ruta_archivo.endswith('.csv'):        
        features_final = pd.read_csv(ruta_archivo, sep=',', encoding='utf-8')
        features_final['ds'] = pd.to_datetime(features_final['ds']).dt.strftime('%Y%m').astype(int)
        features_final.drop(columns=['y','yhat1','yhat2'], inplace=True)
        df_entrada = df_entrada.merge(features_final, on=['periodo', 'product_id'], how='left')
        return df_entrada
    
    # Leer y preparar los datos
    df = pd.read_csv("../../data/preprocessed/base.csv", sep=',')
    df = df.groupby(['periodo', 'product_id'])['tn'].sum().reset_index()
    df['ds'] = pd.to_datetime(df['periodo'], format='%Y%m')
    df.rename(columns={'tn': 'y'}, inplace=True)
    df = df.sort_values(['product_id', 'ds'])

    # df = df[df['periodo'] < 201910]

    # Lista para guardar features
    feature_dfs = []

    # Recorrer productos
    for product_id in tqdm(df['product_id'].unique(), desc="Procesando productos"):
        df_prod = df[df['product_id'] == product_id][['ds', 'y']].copy()
        n_months = len(df_prod)
        
        # Estrategias para series cortas
        if n_months < 14:
            if n_months < 3:  # Series muy cortas (menos de 3 meses)
                # Crear un dataframe sintético con ceros
                min_date = df_prod['ds'].min()
                max_date = df_prod['ds'].max()
                full_range = pd.date_range(start=min_date, end=max_date, freq='M')
                df_prod = df_prod.set_index('ds').reindex(full_range).fillna(0).reset_index()
                df_prod.columns = ['ds', 'y']
                n_months = len(df_prod)
            
            # Ajustar parámetros del modelo para series cortas
            n_lags = min(6, n_months - 2)  # Reducir lags para series cortas
            n_forecasts = min(2, n_months - n_lags - 1)
            
            model = NeuralProphet(
                n_lags=n_lags,
                n_forecasts=n_forecasts,
                yearly_seasonality=True if n_months >= 12 else False,
                weekly_seasonality=False,
                daily_seasonality=False,
                learning_rate=1.0,
                epochs=30  # Reducir épocas para series cortas
            )
            
            # Solo añadir estacionalidad mensual si hay suficientes datos
            if n_months >= 6:
                model = model.add_seasonality(name='monthly', period=30.5, fourier_order=3)
        else:
            # Configuración estándar para series largas
            model = NeuralProphet(
                n_lags=12,
                n_forecasts=2,
                yearly_seasonality=True,
                learning_rate=1.0,
                epochs=50
            )
            model = model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        try:
            # Entrenar (con validación silenciosa para series cortas)
            with np.errstate(divide='ignore', invalid='ignore'):
                model.fit(df_prod, freq='M', progress='none')
            
            # Generar datos históricos
            future = model.make_future_dataframe(df_prod, n_historic_predictions=True)
            forecast = model.predict(future)
            
            # Filtrar columnas útiles
            keep_cols = ['ds', 'trend', 'season_yearly', 'season_monthly', 'autoregressive']
            keep_cols = [col for col in keep_cols if col in forecast.columns]
            forecast = forecast[['ds'] + keep_cols].copy()
            
            forecast['product_id'] = product_id
            forecast['serie_larga'] = n_months >= 14  # Marcar si es serie larga
            
            feature_dfs.append(forecast)
        except Exception as e:
            print(f"Error procesando product_id {product_id}: {str(e)}")
            continue

    # Concatenar todos
    features_final = pd.concat(feature_dfs, ignore_index=True)

    # Exportar
    features_final.to_csv("features_neuralprophet_completo.csv", index=False)
    print(f"✅ Features extraídas correctamente. Procesadas {len(feature_dfs)} series.")
    
    features_final['ds'] = features_final['ds'].dt.strftime('%Y%m').astype(int)  # Convertir a formato YYYYMM
    features_final.rename(columns={'ds': 'periodo'}, inplace=True) 
    df_entrada = df_entrada.merge(features_final, on=['periodo', 'product_id'], how='left')
    
    return df_entrada



def correlacion_exogenas(df):
    # Correlación móvil entre tn_zscore y dolar (ventana de 12 meses)
    df['corr_tn_dolar'] = df['tn_zscore'].rolling(window=12).corr(df['dolar'])

    # Correlación móvil entre tn_zscore e ipc
    df['corr_tn_ipc'] = df['tn_zscore'].rolling(window=12).corr(df['ipc'])
    
    # Correlación entre tn_zscore y dolar para cada producto_id
    # Calcular correlación por grupo
    corr_df = df.groupby('product_id').apply(
        lambda g: g['tn_zscore'].corr(g['dolar'])
    ).reset_index(name='corr_tn_dolar_x_prod')

    # Unir al DataFrame original
    df = df.merge(corr_df, on='product_id', how='left')

    return df


def dwt_features_serie(df, cant_clusters=50):
    """    
    Calcula características adicionales para la serie temporal utilizando DWT.
    """
    
    df = df.sort_values(["product_id", "periodo"])

    # Pivotear para tener series como filas y periodos como columnas
    pivot = df.pivot(index="product_id", columns="periodo", values="tn").fillna(0)

    # Escalar las series temporalmente
    scaler = TimeSeriesScalerMeanVariance()
    series_scaled = scaler.fit_transform(pivot.values)

    # Calcular matriz DTW
    dtw_matrix = cdist_dtw(series_scaled)

    # Clustering con KMeans sobre la matriz de distancia
    n_clusters = cant_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(dtw_matrix)

    # Asignar cluster
    clusters = pd.DataFrame({
        "product_id": pivot.index,
        "dtw_cluster": kmeans.labels_
    })

    # Calcular distancia al centroide por producto
    dist_to_centroid = []
    for i, serie in enumerate(dtw_matrix):
        centroide_idx = np.where(kmeans.labels_ == kmeans.labels_[i])[0]
        centroide_series = dtw_matrix[i, centroide_idx].mean()
        dist_to_centroid.append(centroide_series)
    clusters["dist_to_centroid"] = dist_to_centroid

    # Calcular similitud con productos clave
    top_product_ids = df.groupby("product_id")["tn"].sum().sort_values(ascending=False).head(20).index
    key_products_series = pivot.loc[top_product_ids].values

    similarity_to_top = []
    for i, row in pivot.iterrows():
        similarities = [dtw(row.values, pivot.loc[pid].values) for pid in top_product_ids]
        similarity_to_top.append(np.min(similarities))
    clusters["simil_to_top"] = similarity_to_top

    # Exportar features
    features_dtw = clusters
    features_dtw.to_csv("dtw_features.csv", index=False)
    print("✅ Features DTW generadas y guardadas en 'dtw_features.csv'")

    
    df = df.merge(features_dtw, on="product_id", how="left")

    return df



def descomposicion_serie_temporal(df, col='tn'):
    """
    Descompone la serie temporal en componentes aditivos y multiplicativos,
    con manejo seguro de ceros/negativos para el modelo multiplicativo.
    También crea lags estacionalmente ajustados.
    
    Args:
        df: DataFrame con la serie temporal
        col: Nombre de la columna a descomponer
        
    Returns:
        DataFrame con las nuevas características añadidas
    """
    
    for i in [1, 2, 3, 12]:
        df[f'tn_lag_{i}'] = df.groupby('product_id')['tn'].shift(i)
    
    # Asegurar que no hay valores nulos
    serie = df[col].dropna()
    
    # --- Descomposición ADITIVA ---
    result_add = seasonal_decompose(serie, model='additive', period=12)
    df[f'{col}_trend_decomposed_additive'] = result_add.trend
    df[f'{col}_seasonal_decomposed_additive'] = result_add.seasonal
    df[f'{col}_residual_decomposed_additive'] = result_add.resid
    
    # --- Descomposición MULTIPLICATIVA (con manejo de ceros/negativos) ---
    # Pre-procesamiento para modelo multiplicativo
    serie_mult = serie.copy()
    
    # 1. Manejar ceros (reemplazar con valor pequeño)
    min_positivo = serie[serie > 0].min() / 2  # Mitad del mínimo valor positivo
    serie_mult = np.where(serie <= 0, min_positivo, serie)
    
    # 2. Aplicar descomposición
    result_mult = seasonal_decompose(serie_mult, model='multiplicative', period=12)
    
    # 3. Almacenar resultados
    df[f'{col}_trend_decomposed_multiplicative'] = result_mult.trend
    df[f'{col}_seasonal_decomposed_multiplicative'] = result_mult.seasonal
    df[f'{col}_residual_decomposed_multiplicative'] = result_mult.resid
    
    # --- Interacciones ---
    df[f'{col}_trend'] = df[col].rolling(window=12, min_periods=1).mean()
    
    # Interacción tendencia-estacionalidad
    df[f'{col}_trend_season_interaction_additive'] = df[f'{col}_trend'] * df[f'{col}_seasonal_decomposed_additive']
    df[f'{col}_trend_season_interaction_multiplicative'] = df[f'{col}_trend'] * df[f'{col}_seasonal_decomposed_multiplicative']
    
    # --- Lags estacionalmente ajustados ---
    for i in [1, 2, 3, 12]:
        # Versión aditiva
        seasonal_add = df[f'{col}_seasonal_decomposed_additive']
        df[f'{col}_lag_{i}_season_adj_add'] = df[f'{col}_lag_{i}'] / seasonal_add.replace(0, 1)
        
        # Versión multiplicativa (con manejo seguro)
        seasonal_mult = df[f'{col}_seasonal_decomposed_multiplicative']
        # Reemplazar ceros/negativos con 1 (elemento neutro)
        seasonal_mult_safe = np.where(seasonal_mult <= 0, 1, seasonal_mult)
        df[f'{col}_lag_{i}_season_adj_mul'] = df[f'{col}_lag_{i}'] / seasonal_mult_safe
        
        # Normalización opcional
        df[f'{col}_lag_{i}_season_adj_add_norm'] = (df[f'{col}_lag_{i}_season_adj_add'] - df[f'{col}_lag_{i}_season_adj_add'].mean()) / df[f'{col}_lag_{i}_season_adj_add'].std()
        df[f'{col}_lag_{i}_season_adj_mul_norm'] = (df[f'{col}_lag_{i}_season_adj_mul'] - df[f'{col}_lag_{i}_season_adj_mul'].mean()) / df[f'{col}_lag_{i}_season_adj_mul'].std()
    
    return df



def chatGPT_features_serie(df, col):
    """    
        Calcula características adicionales para la serie temporal de yerba mate.
    """
    df['periodo_dt'] = pd.to_datetime(df['periodo'], format='%Y%m')
    
    ## 5. Características de tendencia y estacionalidad
    df[f'{col}_expanding_mean'] = df[col].expanding().mean()
    df[f'{col}_cumulative_sum'] = df[col].cumsum()

    ## 6. Características de diferencia estacional (12 meses para datos mensuales)
    ventana = [6,12,18,24]
    for v in ventana:
        df[f'{col}_seasonal_diff_{v}'] = df[col].diff(12)

    ## 7. Estadísticas anuales comparativas
    df[f'{col}_vs_prev_year'] = df[col] / df[f'{col}_lag_12'] - 1  # Crecimiento interanual

    ## 8. Componentes de descomposición (simplificada)
    # Tendencia (usando media móvil de 12 meses)
    df[f'{col}_trend'] = df[col].rolling(window=12, min_periods=1).mean()
    # Estacionalidad (diferencia entre valor real y tendencia)
    df[f'{col}_seasonality'] = df[col] - df[f'{col}_trend']

    
    ## 10. Características de aceleración/deceleración
    df[f'{col}_acceleration'] = df[f'{col}_delta_lag_2'].diff(1)  # Cambio en la tasa de cambio
    
    """ 
        Calcula estadísticas de ventana dinámica para la columna col.
    """
    
    # Medias móviles exponenciales
    df[f'{col}_ewm_alpha_0.3'] = df[col].ewm(alpha=0.3, adjust=False).mean()
    df[f'{col}_ewm_alpha_0.5'] = df[col].ewm(alpha=0.5, adjust=False).mean()

    # Medias móviles centradas
    df[f'{col}_rolling_center_mean_3'] = df[col].rolling(window=3, center=True).mean()

    # Sumas acumuladas por año
    df[f'{col}_ytd_sum'] = df.groupby(df['periodo_dt'].dt.year)[col].cumsum()
    
    """    
        Calcula componentes de tendencia y ciclo para la columna col.
    """
     # Modelado de tendencia polinomial
    df[f'{col}_time_index'] = range(len(df))
    df[f'{col}_trend_linear'] = np.poly1d(np.polyfit(df[f'{col}_time_index'], df[col], 1))(df[f'{col}_time_index'])
    df[f'{col}_trend_quadratic'] = np.poly1d(np.polyfit(df[f'{col}_time_index'], df[col], 2))(df[f'{col}_time_index'])

    # Residuales de tendencia
    df[f'{col}_residual_trend'] = df[col] - df[f'{col}_trend_linear']
    

    """    
        Calcula características de cambio de régimen para la columna col.
    """
    # Z-Score respecto a ventana móvil
    df[f'{col}_zscore_6'] = (df[col] - df[f'{col}_rolling_mean_6']) / df[f'{col}_rolling_std_6']

    # Detección de outliers
    df[f'{col}_is_outlier_3sigma'] = np.where(np.abs(df[f'{col}_zscore_6']) > 3, 1, 0)

    # Cambios bruscos (spikes)
    df[f'{col}_spike_up'] = np.where(df[f'{col}_delta_lag_2'] > df[f'{col}_rolling_std_3'], 1, 0)
    df[f'{col}_spike_down'] = np.where(df[f'{col}_delta_lag_2'] < -df[f'{col}_rolling_std_3'], 1, 0)
    
    """
        Calcula características de cambio de régimen para la columna col.
        Esta función incluye métodos ingenuos y de promedio móvil para pronósticos.
    """
    # Método ingenuo (último valor)
    df[f'{col}_naive_forecast'] = df[col].shift(1)

    # Seasonal naive (valor del mismo período año anterior)
    df[f'{col}_seasonal_naive'] = df[col].shift(12)

    # Promedio móvil como forecast
    df[f'{col}_ma_forecast_3'] = df[f'{col}_rolling_mean_3'].shift(1)
    
    
    """    
        Calcula estadísticas de ventanas asimétricas para la columna col.
    """
   
    # Mejor mes histórico
    df[f'{col}_best_month_rank'] = df.groupby('month')[col].rank(ascending=False)

    # Comparación con mismo mes año anterior
    df[f'{col}_vs_last_year_same_month'] = df[col] / df[f'{col}_lag_12'] - 1

    # Acumulado últimos 3 vs mismos 3 meses año anterior
    df[f'{col}_last3_vs_ly3'] = (df[col] + df[f'{col}_lag_1'] + df[f'{col}_lag_2']) / (df[f'{col}_lag_12'] + df[f'{col}_lag_13'] + df[f'{col}_lag_14']) - 1
    
    
    df.drop(columns=['periodo_dt'], inplace=True)  # Eliminar la columna temporal
    
    return df








