import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from optuna.storages import RDBStorage
from optuna.artifacts import FileSystemArtifactStore, upload_artifact
import os
import json



def optimizar_con_optuna(train):
    """
    Optimiza los hiperparámetros de un modelo LightGBM utilizando Optuna.
    """
    # Asegurarse de que 'periodo' esté en formato datetime
    train = train.sort_values("periodo")  # o la columna de fecha

    
    # Eliminar columnas no necesarias
    datetime_cols = train.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    # cols_to_drop = ['target', 'periodo'] + datetime_cols  # Asegúrate de incluir 'periodo'
    X = train.drop(columns=[*datetime_cols, 'target'])
    y = train['target']

    # Sample Weights (ej: ponderar por toneladas históricas)
    # sample_weight = train['tn_zscore'].values if 'tn_zscore' in train.columns else None
   
    # ---------------------------------------------------
   
    tscv = TimeSeriesSplit(
        n_splits=5,
        test_size=1,  # Validar 1 mes (el mes+2 desde el último mes de entrenamiento)
        gap=1         # Respetar el mes intermedio (ej: entrenar hasta 201806, predecir 201808)
    )

    # ---------------------------------------------------
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 15, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),  # Log-scale para LR pequeñas
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),  # Frecuencia de bagging
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),  # Log-scale para regularización
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 15),  # Profundidad máxima
            'max_bin': trial.suggest_int('max_bin', 100, 500),  # Optimizar bins: 100 a 255
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
            'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),  # Alternativa a GBDT
            'seed': 42,
            'verbosity': -1
        }
        
        rmse_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Sample Weights para el fold actual
            # if sample_weight is not None:
            #     sample_weight_fold = sample_weight[train_idx]
            # else:
            #     sample_weight_fold = None
            
            train_data = lgb.Dataset(
                X_train_fold, 
                label=y_train_fold,
                # weight=sample_weight_fold  # Aplicar sample_weight
            )
            val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )
            
            y_pred = model.predict(X_val_fold)
            rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
            rmse_scores.append(rmse)
        
        return np.mean(rmse_scores)

    
    # ---------------------------------------------------
    def print_best_trial(study, trial):
        print(f"Mejor trial hasta ahora: RMSE={study.best_value:.4f}, Parámetros={study.best_trial.params}")

    study = optuna.create_study(direction='minimize') # Minimizar RMSE
    study.optimize(
        objective,
        n_trials=50,  # Aumentar trials para búsqueda exhaustiva
        callbacks=[print_best_trial],
        timeout=3600  # Límite de tiempo opcional (1 hora)
    )

    print("Mejores hiperparámetros:", study.best_params)
    # ---------------------------------------------------
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1
    })

    
    # Dataset completo con sample_weight
    # final_train_data = lgb.Dataset(
    #     X, 
    #     label=y,
    #     # weight=sample_weight  # Aplicar sample_weight global
    # )

    # Entrenar con early stopping en un pequeño holdout (opcional)
    # final_model = lgb.train(
    #     best_params,
    #     final_train_data,
    #     num_boost_round=1000,
    #     callbacks=[lgb.log_evaluation(50)]
    # )

    # Guardar modelo
    # final_model.save_model('modelo_final_lightgbm.txt')
    guardar_hiperparametros(best_params, "lgb_v1")




def optimizar_con_optuna_con_semillerio(train, semillas=[42, 101, 202, 303, 404], version="v1"):
    """
    Optimiza los hiperparámetros de un modelo LightGBM utilizando Optuna,
    aplicando semillerío durante la evaluación de cada trial.
    """

    datetime_cols = train.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    X = train.drop(columns=[*datetime_cols, 'target'])
    y = train['target']

    tscv = TimeSeriesSplit(
        n_splits=5,
        test_size=1,
        gap=1
    )

    def objective(trial):
        base_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 15, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'max_bin': trial.suggest_int('max_bin', 100, 500),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
            'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
            'verbosity': -1
        }

        rmse_scores_fold = []

        for train_idx, val_idx in tscv.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            rmse_seeds = []

            for seed in semillas:
                params = base_params.copy()
                params['seed'] = seed

                train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
                val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=1000,
                    valid_sets=[val_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(0)
                    ]
                )

                y_pred = model.predict(X_val_fold)
                rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
                rmse_seeds.append(rmse)

            # Promedio de RMSE para este fold
            rmse_scores_fold.append(np.mean(rmse_seeds))

        return np.mean(rmse_scores_fold)

    def print_best_trial(study, trial):
        print(f"Mejor trial hasta ahora: RMSE={study.best_value:.4f}, Parámetros={study.best_trial.params}")

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, callbacks=[print_best_trial], timeout=3600)

    print("Mejores hiperparámetros:", study.best_params)

    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1
    })

    guardar_hiperparametros(best_params, version)


    
 
 
def semillerio_en_prediccion(train, test, version="v1"):
    """
    Entrena un modelo LightGBM con múltiples semillas y promedia las predicciones. 
    """
    
    datetime_cols = train.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    X_train = train.drop(columns=[*datetime_cols, 'target'])
    y_train = train['target']    
    X_test = test.drop(columns=[*datetime_cols, 'target'])
    train_data = lgb.Dataset(X_train, label=y_train)

    
    # Número de repeticiones con semillas distintas
    best_params = levantar_hiperparametros(version)
    
    # Cargar datos de entrenamiento y prueba
    seeds = [42, 101, 202, 303, 404]
    predictions = []

    # Lista para almacenar los feature importance de cada modelo
    feature_importances = []
    feature_names = X_train.columns.tolist()  # Nombres de las features
    

    for seed in seeds:
        params = best_params.copy()
        params['seed'] = seed
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_test)
        predictions.append(y_pred)
        
        # Obtener feature importance para este modelo
        importance = model.feature_importance(importance_type='gain')  # 'gain' o 'split'
        feature_importances.append(importance)

    # Promediar predicciones
    final_prediction = np.mean(predictions, axis=0)   
    
    # Crear DataFrame con IDs y predicciones
    result_df = test[['periodo', 'product_id', 'target']].copy()
    result_df['pred'] = final_prediction
    
    # Procesar y guardar feature importance
    #############################################
    # 1. Promediar los feature importance de todos los modelos
    avg_importance = np.mean(feature_importances, axis=0)
    
    # 2. Crear DataFrame con los resultados
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)
    
    # 3. Guardar a CSV
    # importance_df.to_csv('feature_importance.csv', index=False)
    
    # 4. Guardar a JSON (opcional)
    importance_dict = importance_df.set_index('feature')['importance'].to_dict()
    with open(f'./feature_importance/{version}.json', 'w') as f:
        json.dump(importance_dict, f, indent=4)
    #############################################
    
    return result_df
    
    
    
def semillerio_en_prediccion_v2(params, train_data, val_data, X_test):
    """
    Entrena un modelo LightGBM con múltiples semillas y promedia las predicciones.  
    Esto ayuda a reducir la varianza y mejorar la robustez del modelo.
    """
    n_seeds = 10
    predictions = []

    for seed in range(n_seeds):
        params['seed'] = seed  # LightGBM usa 'seed' en lugar de 'random_state'
        model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[val_data])
        predictions.append(model.predict(X_test))

    # Predicción final promediada
    final_pred = np.mean(predictions, axis=0)
        
        

def semillerio_en_entrenamiento(params, train_data, X_test):
    
    for seed in [42, 123, 456]:  # Semillas fijas
        params['seed'] = seed
        model = lgb.train(params, train_data, num_boost_round=1000)
        pred = model.predict(X_test)
        # Guardar pred para cada semilla y promediar después
        
        
        
def guardar_hiperparametros(best_params, name='lgb_v1'):
    """
    Guarda los mejores hiperparámetros en un archivo JSON.
    """
    # Guardar best_params en un archivo JSON
    with open(f'./best_params/{name}.json', 'w') as f:
        json.dump(best_params, f, indent=4)


def levantar_hiperparametros(nombre):
    """
    Levanta los hiperparámetros guardados en un archivo JSON.
    
    Args:
        nombre (str): Nombre del archivo (sin extensión .json).
    
    Returns:
        dict: Diccionario con los hiperparámetros. None si hay error.
    """
    try:
        with open(f'./best_params/{nombre}.json', 'r') as f:
            best_params = json.load(f)  # ¡Usar json.load() en lugar de json.dump()!
        return best_params
    except FileNotFoundError:
        print(f"Error: Archivo './best_params/{nombre}.json' no encontrado.")
        return None
    except json.JSONDecodeError:
        print(f"Error: El archivo './best_params/{nombre}.json' no tiene formato JSON válido.")
        return None


def optimizar_con_optuna_SQLITE(train, path_to_db="sqlite:///optuna_lgb.db", study_name="LGBIvanModeloCompleto"):
    """
    Optimiza los hiperparámetros de un modelo LightGBM utilizando Optuna y guarda resultados en SQLite.

    Se utiliza así:
    best_params, study = optimizar_con_optuna(df, path_to_db="sqlite:///mi_lgb.db", study_name="LGBVentas36M")
    """

    # Filtramos columnas datetime
    datetime_cols = train.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    X = train.drop(columns=datetime_cols)
    y = train['target']

    tscv = TimeSeriesSplit(n_splits=5, test_size=1, gap=1)

    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 15, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'max_bin': trial.suggest_int('max_bin', 100, 500),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
            'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
            'seed': 42,
            'verbosity': -1
        }

        rmse_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )

            y_pred = model.predict(X_val_fold)
            rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
            rmse_scores.append(rmse)

        # Guardar RMSE promedio como user_attr en la base de datos
        trial.set_user_attr("cv_rmse", np.mean(rmse_scores))
        return np.mean(rmse_scores)

    def print_best_trial(study, trial):
        print(f"Mejor trial hasta ahora: RMSE={study.best_value:.4f}, Parámetros={study.best_trial.params}")

    # ----------------------------- NUEVO: Configurar almacenamiento -----------------------------
    storage = optuna.storages.RDBStorage(url=path_to_db)
    study = optuna.create_study(
        direction='minimize',
        storage=storage,
        study_name=study_name,
        load_if_exists=True
    )
    # -------------------------------------------------------------------------------------------

    # ----------------------------- OPCIONAL: Artefactos en carpeta local -----------------------
    PATH_TO_OPTUNA_ARTIFACTS = "./optuna_artifacts"
    artifact_store = FileSystemArtifactStore(base_path=PATH_TO_OPTUNA_ARTIFACTS)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # -------------------------------------------------------------------------------------------

    study.optimize(
        objective,
        n_trials=50,
        callbacks=[print_best_trial],
        timeout=3600
    )

    print("Mejores hiperparámetros:", study.best_params)
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1
    })

    return best_params, study




def desnormalizar(df):
    """ 
    Desnormaliza las predicciones y los targets de un DataFrame.
    """
    # Levanto los 780 productos
    products_ok = pd.read_csv("../../data/raw/product_id_apredecir201912.csv")
    productds_ok = products_ok['product_id'].unique()
    
    # Filtrar los 780 productos
    df = df[df['product_id'].isin(productds_ok)]
    
    
    # desnormalizar las predicciones
    df_stats = pd.read_csv("./datasets/target_stats_201909.csv", sep=',') 
    df = df.merge(df_stats, on='product_id', how='left')
    
    df['target'] = df['target'] * df['target_std'] + df['target_mean']
    df['pred'] = df['pred'] * df['target_std'] + df['target_mean']
    
    df.drop(columns=['target_mean', 'target_std'], inplace=True)
    
    return df
    
    
    
def total_forecast_error(df):
    """ Calcula el Total Forecast Error (TFE) entre las predicciones y los targets de un DataFrame.
    """
    numerador = (df['target'] - df['pred']).abs().sum()
    denominador = df['target'].sum()
    
    return numerador / denominador