import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier 

mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment(experiment_id='1')

models = {
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'params': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 4, 5],
            'min_samples_leaf': [2, 5, 10, 20],
            'max_features': ['sqrt', 'log2',None]
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [3, 4, 5],
            'min_samples_leaf': [2, 5, 10, 20],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced_subsample',"balanced"]
        }
    },
    'XGBClassifier':{
        'model': XGBClassifier(random_state=42,objective='binary:logistic',eval_metric='auc',scale_pos_weight=4),
        'params': {
            'n_estimators': [300, 500],
            'max_depth': [3, 4],
            'learning_rate': [0.03, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    }
}

def build_pipeline(preprocessar, X_treino, y_treino):
    resultados = []

    for nome, config in models.items():

        with mlflow.start_run(run_name=nome) as run:
            pipeline = Pipeline(
                steps=[
                    ("preprocessar", preprocessar),
                    ("model", GridSearchCV(config['model'], config['params'], cv=5, scoring='recall', n_jobs=4))
                    ])
        
            run_id = run.info.run_id
            pipeline.fit(X_treino,y_treino)

            model = pipeline.named_steps["model"].best_estimator_
            bestparamsmodel = pipeline.named_steps["model"].best_params_
            feature_names = pipeline.named_steps["preprocessar"].get_feature_names_out()
            resultados.append({'nome': nome, 'model': model, 'pipeline': pipeline, 'features': feature_names, 'id_run': run_id})
            print(f'{nome} melhores parâmetros: {bestparamsmodel}')

    return resultados