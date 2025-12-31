from preprocessing import preprocessing, loading_data
from modeling import build_pipeline, models
from performance import get_performance

df = loading_data()

def run(drop_col=[], complain=''): 
    X_treino, X_teste, y_treino, y_teste, preprocessar = preprocessing(df,drop_col)
    resultados = build_pipeline(preprocessar, X_treino, y_treino,complain=complain)
    get_performance(resultados,X_treino,y_treino,X_teste,y_teste)

def main():
    run()
    run(drop_col=['Complain'], complain=' no Complain')

if __name__ == "__main__":
    main()