import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import TimeSeriesSplit
from optuna.storages import RDBStorage
from optuna.artifacts import FileSystemArtifactStore, upload_artifact
import os
import json



def optimizar_con_optuna(train, version="v1"):
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
            # rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
            rmse = mean_squared_error(y_val_fold, y_pred)
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
    guardar_hiperparametros(best_params, version)




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
                # rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
                rmse = mean_squared_error(y_val_fold, y_pred)
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
        
    return final_pred    

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



#### Más Rústico #################################################

def entrenar_rustico(df_train, df_val, df_test, features):
    
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'random_state': 42
        }

        X = df_train.drop(columns=['target'])
        y = df_train['target']
        # sw = df_train['tn']
        # sw = df_train['tn'].clip(lower=1e-5)
        sw = df_train['tn_scaled']

        tscv = TimeSeriesSplit(n_splits=5, max_train_size=24 , test_size=2 )
        maes = []
        for train_idx, val_idx in tscv.split(X):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            sw_train_fold = sw.iloc[train_idx]

            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train_fold, y_train_fold,
                sample_weight=sw_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                eval_metric="mae",
                callbacks=[
                    early_stopping(stopping_rounds=50),
                    log_evaluation(0)  # cambiar a 100 si querés ver progreso cada 100 iteraciones
                ]
            )
            pred = model.predict(X_val_fold)
            maes.append(mean_absolute_error(y_val_fold, pred))

        return np.mean(maes)


    # Crear un estudio de Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)


    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'random_state': 42
    })

    # Imprimir los mejores hiperparámetros
    print("Mejores hiperparámetros: ", study.best_params)
    print("Mejor MAE: ", study.best_value)

    model_val = lgb.LGBMRegressor(**best_params)
    model_val.fit(
        df_train[features], df_train['target'],
        sample_weight=df_train['tn_scaled']
    )

    y_val_pred = model_val.predict(df_val[features])
    mae_val = mean_absolute_error(df_val['target'], y_val_pred)
    print(f"MAE en validación externa (201909–201910): {mae_val:.4f}")
    
    df_fit = pd.concat([df_train, df_val])

    model_final = lgb.LGBMRegressor(**best_params)
    model_final.fit(
        df_fit[features], df_fit['target'],
        sample_weight=df_fit['tn_scaled']
    )

    y_test_pred = model_final.predict(df_test[features])
    mae_test = mean_absolute_error(df_test['target'], y_test_pred)
    print(f"MAE en test final (201911–201912): {mae_test:.4f}")









##### Más PRO ####################################################

def evaluate_model(df, periodos_train, periodos_val, features, params, 
                   seeds=[42, 123, 456], use_cv=True, n_splits=5, verbose=False):
    """
    Evalúa un modelo con múltiples semillas. Usa CV temporal interna si use_cv=True.
    """
    df_train = df[df['periodo'].isin(periodos_train)].copy()
    df_val = df[df['periodo'].isin(periodos_val)].copy()
    
    # Validaciones básicas
    if df_train.empty or df_val.empty:
        raise ValueError("Train o Val está vacío.")

    

    X_train = df_train[features]
    y_train = df_train['target']
    ### Peso
    sw_train = df_train['tn_scaled']
    # df_train['tn'] = pd.to_numeric(df_train['tn'], errors='coerce').fillna(1e-5).clip(lower=1e-5)

    X_val = df_val[features]
    y_val = df_val['target']

    if use_cv:
        cv_maes = []
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            fold_maes = []

            for seed in seeds:
                model = lgb.LGBMRegressor(**params, random_state=seed)
                model.fit(
                    X_train.iloc[train_idx], y_train.iloc[train_idx],
                    sample_weight=sw_train.iloc[train_idx],
                    eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                    eval_metric="mae",
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
                )
                pred = model.predict(X_train.iloc[val_idx])
                fold_maes.append(mean_absolute_error(y_train.iloc[val_idx], pred))

            cv_maes.append(np.mean(fold_maes))
        
        mae_cv = np.mean(cv_maes)
        if verbose:
            print(f"CV MAE: {mae_cv:.4f}")
    else:
        mae_cv = None

    # Entrenamiento final con todo el set de train
    final_preds = []
    models = []

    for seed in seeds:
        model = lgb.LGBMRegressor(**params, random_state=seed)
        model.fit(
            X_train, y_train,
            sample_weight=sw_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )
        pred = model.predict(X_val)
        final_preds.append(pred)
        models.append(model)

    ensemble_pred = np.mean(final_preds, axis=0)
    mae_val = mean_absolute_error(y_val, ensemble_pred)

    if verbose:
        print(f"Ensemble MAE Val: {mae_val:.4f}")

    return mae_cv if use_cv else mae_val, models, ensemble_pred



def create_objective(df, periodos_train, periodos_val, features, use_cv=True):
    """ 
    Crea una función objetivo para Optuna que evalúa un modelo LightGBM.
    Esta función se utiliza para la optimización de hiperparámetros.
    """
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_samples': trial.suggest_int('min_child_weight', 1, 10) * 5,
            'bagging_fraction': trial.suggest_float('subsample', 0.5, 1.0),
            'bagging_freq': 1,
            'feature_fraction': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'deterministic': True,
            'force_col_wise': True
            # 'linear_tree': True ##### CUIDADO!!!
        }

        try:
            mae, _, _ = evaluate_model(df, periodos_train, periodos_val, features, params, use_cv=use_cv)
            return mae
        except Exception as e:
            print(f"Trial error: {e}")
            return float("inf")
    return objective




def run_bayesian_optimization(df, periodos_train, periodos_val, periodos_test,
                              n_trials=20, use_cv=True):
    """ 
    Ejecuta la optimización bayesiana de hiperparámetros para LightGBM.
    
    """
    features = [col for col in df.columns if col not in ['target']]
    print(f"Features: {features}")

    objective = create_objective(df, periodos_train, periodos_val, features, use_cv=use_cv)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params.update({
        # 'min_child_samples': best_params.pop('min_child_weight') * 5,
        # 'bagging_fraction': best_params.pop('subsample'),
        # 'bagging_freq': 1,
        # 'feature_fraction': best_params.pop('colsample_bytree'),
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'deterministic': True,
        'force_col_wise': True
        # 'linear_tree': True ##### CUIDADO!!!
    })

    print("\n🏆 Mejores parámetros:")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    # Entrenamiento final
    print("\nEntrenando modelo final...")
    mae_val, models, _ = evaluate_model(df, periodos_train, periodos_val, features, best_params, use_cv=False, verbose=True)

    # Evaluación en test
    df_test = df[df['periodo'].isin(periodos_test)].copy()
    if df_test.empty:
        print("No hay datos de test")
        return study, models, None

    X_test = df_test[features]
    y_test = df_test['target']

    preds = [model.predict(X_test) for model in models]
    ensemble_pred = np.mean(preds, axis=0)
    mae_test = mean_absolute_error(y_test, ensemble_pred)

    print(f"\n🎯 MAE Test Ensemble: {mae_test:.4f}")

    return study, models, {
        'mae_val': mae_val,
        'mae_test': mae_test,
        'ensemble_pred': ensemble_pred,
        'best_params': best_params
    }



def main_entrenar(df, VERSION="v12"):
    """ 
    Entrena un modelo LightGBM con optimización bayesiana y semillerío.
    """
    scaler = MinMaxScaler(feature_range=(1, 100))
    df['tn_scaled'] = scaler.fit_transform(df[['tn']])
    
    # Configurar períodos
    periodos_train  = [ 201701, 201702, 201703, 201704, 201705, 201706, 201707, 201708, 201709, 201710, 201711, 201712,
                        201801, 201802, 201803, 201804, 201805, 201806, 201807, 201808, 201809, 201810, 201811, 201812,
                        201901, 201902, 201903, 201904, 201905, 201906, 201907, 201908, 201909, 201910  ]
    periodos_val    = [ 201907, 201908  ]
    periodos_test   = [ 201909, 201910  ]

    # Ejecutar optimización
    study, models, results = run_bayesian_optimization(
        df=df,
        periodos_train=periodos_train,
        periodos_val=periodos_val,
        periodos_test=periodos_test,
        n_trials=50,
        use_cv=True  # True = validación cruzada temporal dentro de train
    )

    # Análisis adicional
    print("\nTop 5 mejores trials:")
    top_trials = study.trials_dataframe().nsmallest(5, 'value')
    for i, (_, row) in enumerate(top_trials.iterrows(), 1):
        print(f"{i}. MAE: {row['value']:.4f}")

    guardar_hiperparametros(study.best_params, VERSION)
    
    return study, models, results





def predict_with_ensemble(models, df_future, features):
    """
    Realiza predicción con ensemble (promedio) sobre un nuevo dataset.
    """
    if df_future.empty:
        raise ValueError("El dataframe de predicción está vacío.")

    X_future = df_future[features]

    # Promedio de predicciones de todas las semillas
    preds = [model.predict(X_future) for model in models]
    ensemble_pred = np.mean(preds, axis=0)

    return ensemble_pred




######################################################

def promedio_12_meses_780p():
    
    df = pd.read_csv("./datasets/periodo_x_producto_con_target.csv", sep=',', encoding='utf-8')
    df = df[df['periodo'] >= 201901]  # Filtrar desde 201901
    
    productos_ok = pd.read_csv("https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/product_id_apredecir201912.txt", sep="\t")

    df = df.merge(productos_ok, on='product_id', how='inner')
    
    df = df.groupby('product_id').agg({'tn': 'mean'}).reset_index()
    
    return df
    

def feature_importance(df, models, nro_experimento):
    
    df_aux = df.drop(columns=['target'])


    models_aux = models.copy()
    count = 1

    for model in models_aux:
        
        
        feature_importances = pd.DataFrame({
            'feature': df_aux.columns,
            'importance':  model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        
        
        feature_importances.to_csv(f"./feature_importance/exp{nro_experimento}_{count}.csv", index=False, sep=',')
        count += 1
    
    
def feature_importance_promedio(models, features):
    """
    Versión que maneja modelos con diferente número de características
    """
    if not models:
        raise ValueError("La lista de modelos está vacía")
    
    # Crear un diccionario para acumular importancias por nombre de feature
    importance_dict = {feature: 0.0 for feature in features}
    model_count = 0
    
    for model in models:
        try:
            # Obtener nombres e importancias según el modelo
            if hasattr(model, 'feature_importances_'):
                model_features = features  # Asumiendo que son las mismas
                model_imp = model.feature_importances_
            elif hasattr(model, 'get_booster'):  # Para XGBoost
                model_features = model.get_booster().feature_names
                model_imp = model.feature_importances_
            else:
                continue
                
            # Normalizar importancias
            model_imp = model_imp / model_imp.sum()
            
            # Acumular por nombre de feature
            for feature, imp in zip(model_features, model_imp):
                if feature in importance_dict:
                    importance_dict[feature] += imp
            
            model_count += 1
            
        except Exception as e:
            print(f"Error procesando modelo: {str(e)}")
            continue
    
    if model_count == 0:
        raise ValueError("Ningún modelo válido encontrado")
    
    # Calcular promedio
    importance_dict = {k: v/model_count for k, v in importance_dict.items()}
    
    # Crear DataFrame
    importance_df = pd.DataFrame({
        'feature': importance_dict.keys(),
        'importance': importance_dict.values()
    }).sort_values('importance', ascending=False)
    
    return importance_df
