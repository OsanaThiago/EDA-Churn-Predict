from transform.preprocessing import preprocessing, loading_data
from modelling.modeling import build_pipeline, models
from performance.performance import get_performance

df = loading_data()

def run(): 
    X_treino, X_teste, y_treino, y_teste, preprocessar = preprocessing(df)
    resultados = build_pipeline(preprocessar, X_treino, y_treino)
    get_performance(resultados, X_treino, y_treino, X_teste, y_teste)

if __name__ == "__main__":
    run()