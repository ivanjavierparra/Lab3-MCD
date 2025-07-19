# labo3-2025v-final
Repositorio para la Materia Laboratorio de Implementación III de la Maestría en Ciencia de Datos de la Universidad Austral - 2025v

##### Alumno
- Iván Parra

##### Carpetas
- data: contiene las datasets de la competencia y los generados por nosotros.
  - raw: contiene los datasets de la competencia.
  - preprocessed: contiene todos los datasets generados por nosotros. 
- models: contiene modelos guardados de los experimentos.
- notebooks: contiene todos los notebooks de los experimentos.
  - entregable: contiene la mayor parte de los experimentos, principalmente con  
  - exploración: contiene el EDA.
  - feature_engineering: contiene algunas funciones para generar features.
  - model_arima: experimentos con ARIMA.
  - model_autogluon: experimentos
  - model_ensemble
  - model_lgb
  - model_linear_regression
  - model_lstm
  - model_mlforecast
  - model_neural_prophet
  - model_prophet
  - model_xgboost
- outputs: salidas de los experimentos
- scripts: código genérico usado para crear datasets, promediar csv, etc.

# Generación de Datasets
##### Dataset "base.csv"
Para generar este dataset que se usa en muchos experimentos hay que ejecutar el notebook: ./scripts/generador_dataset_sellin.ipynb

##### Dataset "periodo_x_producto_con_target.csv"
Para generar este dataset hay que ejecutar los siguientes notebook en orden:
1. notebooks\entregable\dataset.ipynb
2. notebooks\entregable\target.ipynb

##### Dataset "periodo_x_producto_con_target_transformado.csv"
Para generar este dataset hay que ejecutar los siguientes notebook en orden:
1. notebooks\entregable\dataset.ipynb
2. notebooks\entregable\target.ipynb
3. notebooks\entregable\preprocesamiento.ipynb
   


