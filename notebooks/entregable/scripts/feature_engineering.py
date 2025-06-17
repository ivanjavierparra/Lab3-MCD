import pandas as pd
import numpy as np
from calendar import monthrange
import gc
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
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
    
    date_col = 'periodo'
    product_col = 'product_id'
    target_col = 'tn_zscore'


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
        año = row['año_1']
        mes = row['mes_1']
        
        # Feriados fijos (sin año específico)
        for dia, mes_feriado in feriados:
            if mes == mes_feriado:
                return 1
        
        # Feriados móviles (Semana Santa por año)
        if año == 2017 and mes == 4:  # Abril 2017 (13-16/4)
            return 1
        elif año == 2018 and mes == 3:  # Marzo 2018 (29/3 - 1/4)
            return 1
        elif año == 2019 and mes == 4:  # Abril 2019 (18-21/4)
            return 1
        
        return 0

    # Aplicar la función y crear columna
    df['contiene_feriado'] = df.apply(tiene_feriado, axis=1)