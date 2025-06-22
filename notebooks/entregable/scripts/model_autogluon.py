import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor



def entrenar_y_predecir(df, periodo_limite=201910):
    """
    Entrena un modelo de predicción de series temporales con variables exógenas y realiza predicciones.
    Divide los datos en train/test según el periodo límite.
    
    Args:
        df: DataFrame con los datos históricos
        periodo_limite: Mes a partir del cual se consideran datos de test (formato YYYYMM)
    """
    # Procesamiento inicial
    dfg = df.groupby(['periodo', 'product_id']).agg({'tn': 'sum'}).reset_index()

    # Convertir periodo a datetime
    dfg['periodo_dt'] = pd.to_datetime(dfg['periodo'].astype(str), format='%Y%m')
    dfg.rename(columns={'tn': 'target', 'product_id':'item_id', 'periodo_dt': 'timestamp'}, inplace=True)

    # Filtrar productos a predecir
    productos_ok = pd.read_csv('../../data/raw/product_id_apredecir201912.csv', sep=',')
    dfg = dfg[dfg['item_id'].isin(productos_ok['product_id'].unique())]

    # Cargar y procesar variables exógenas
    dolar = pd.read_csv("./datasets/dolar_oficial_bna.csv", sep=';', encoding='utf-8')
    ipc = pd.read_csv("./datasets/ipc.csv", sep=';', encoding='utf-8')

    # Convertir y unir variables exógenas
    dolar['periodo_dt'] = pd.to_datetime(dolar['periodo'].astype(str), format='%Y%m')
    ipc['periodo_dt'] = pd.to_datetime(ipc['periodo'].astype(str), format='%Y%m')

    # Unir variables exógenas
    dfg = dfg.merge(
        dolar,
        left_on='timestamp', right_on='periodo_dt', how='left'
    )

    dfg = dfg.merge(
        ipc,
        left_on='timestamp', right_on='periodo_dt', how='left'
    )

    # Dividir en train y test
    train = dfg[dfg['periodo'] < periodo_limite].copy()
    test = dfg[dfg['periodo'] >= periodo_limite].copy()
    
    # Columnas a usar
    columns_to_use = ['item_id', 'timestamp', 'target', 'dolar', 'ipc']
    train = train[columns_to_use]
    test = test[columns_to_use]

    # Crear TimeSeriesDataFrame para entrenamiento
    train_data = TimeSeriesDataFrame.from_data_frame(
        train,
        id_column="item_id",
        timestamp_column="timestamp"
    )

    # Configurar y entrenar predictor
    predictor = TimeSeriesPredictor(
        target='target',
        prediction_length=2,  # Predecir 2 meses hacia adelante
        freq="M",        
        known_covariates_names=["dolar", "ipc"]
    ).fit(
        train_data,
        num_val_windows=2,
        presets="medium_quality"
    )
    
    # Preparar datos futuros de variables exógenas
    # Usamos los valores REALES del periodo a predecir (de test)
    future_covariates = test[['timestamp', 'dolar', 'ipc', 'item_id']].copy()
    
    # Convertir a TimeSeriesDataFrame
    future_covariates = TimeSeriesDataFrame.from_data_frame(
        future_covariates,
        id_column="item_id",
        timestamp_column="timestamp"
    )

    # Realizar predicciones
    predictions = predictor.predict(train_data, known_covariates=future_covariates)

    # Procesamiento de resultados
    predictions_v1 = predictions.reset_index()
    predictions_v1 = predictions_v1[["item_id", "timestamp", "mean"]]
    
    # Filtrar la última predicción (mes +2)
    predictions_v1 = predictions_v1[predictions_v1.timestamp == predictions_v1.timestamp.max()]
    
    # Unir con los valores reales para evaluación
    test_eval = test[test['periodo'] == periodo_limite + 2][['item_id', 'target']]  # Datos reales de dic-2019
    predictions_v1 = predictions_v1.merge(
        test_eval,
        on='item_id',
        how='left'
    )
    
    predictions_v1 = predictions_v1.rename(columns={
        "item_id": "product_id",
        "mean": "tn_pred",
        "target": "tn_real"
    })

    # Guardar resultados
    predictions_v1.to_csv("./outputs/prediccion_autogluon_2ventanas.csv", sep=",", index=False)

    return predictions_v1