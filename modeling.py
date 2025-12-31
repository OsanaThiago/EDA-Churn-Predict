import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 

mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment(experiment_id='1')

models = {
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'params': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 5, 10],
            'max_features': ['sqrt', 'log2']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [4, 5, 6, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 5],
            'max_features': ['sqrt', 'log2']
        }
    },
    'RegressaoLogistica': {
        'model': LogisticRegression(random_state=42,class_weight='balanced',solver='saga',max_iter=2000),
        'params': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['saga','lbfgs']
        }
    }
}


def build_pipeline(preprocessar, X_treino, y_treino,complain=''):
    resultados = []

    for nome, config in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessar", preprocessar),
                ("model", GridSearchCV(config['model'], config['params'], cv=5, scoring='recall', n_jobs=4))
                ])
        
        with mlflow.start_run(run_name=nome+complain) as run:
            mlflow.sklearn.autolog()
            run_id = run.info.run_id
            pipeline.fit(X_treino,y_treino)

            model = pipeline.named_steps["model"].best_estimator_
            bestparamsmodel = pipeline.named_steps["model"].best_params_
            feature_names = pipeline.named_steps["preprocessar"].get_feature_names_out()

            # if isinstance(model, (DecisionTreeClassifier,RandomForestClassifier)):
            #     importancia = pd.Series(model.feature_importances_, index=feature_names).reset_index(name="importancia").sort_values("importancia")
            #     importancia.plot(kind="barh", x="index", y="importancia", legend=False, figsize=(8,6))
            #     plt.title(f"Features mais importantes para Churn na {nome}")

            # else:
            #     importancia = pd.Series(model.coef_.ravel(),index=feature_names).reset_index(name="importancia").sort_values("importancia")
            #     importancia.plot(kind="barh", x="index", y="importancia", legend=False, figsize=(8,6))
            #     plt.title("Coeficiente da Regressão Logística")
            #     plt.axvline(0, linestyle='--', color='black')

            resultados.append({'nome': nome, 'model': model, 'pipeline': pipeline, 'features': feature_names, 'id_run': run_id})
            print(f'{nome} melhores parâmetros: {bestparamsmodel}')
    return resultados