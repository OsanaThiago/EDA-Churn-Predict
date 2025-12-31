import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def get_performance(resultados,X_treino,y_treino,X_teste,y_teste):
    for resultado in resultados:

        nome = resultado['nome']
        pipeline = resultado['pipeline']
        id_run = resultado['id_run']

        with mlflow.start_run(run_id=id_run):
            y_treino_predict = pipeline.predict(X_treino)
            y_treino_proba = pipeline.predict_proba(X_treino)[:,1]

            auc_treino = roc_auc_score(y_treino, y_treino_proba)
            roc_treino = roc_curve(y_treino, y_treino_proba)
            cr_treino = classification_report(y_treino, y_treino_predict, target_names=['Não churn','Churn'])

            y_teste_predict = pipeline.predict(X_teste)
            y_teste_proba = pipeline.predict_proba(X_teste)[:,1]

            auc_teste = roc_auc_score(y_teste, y_teste_proba)
            roc_teste = roc_curve(y_teste, y_teste_proba)
            cr_teste = classification_report(y_teste, y_teste_predict, target_names=['Não churn','Churn'])

            mlflow.log_metrics({
                "auc_treino": auc_treino,
                "auc_teste": auc_teste
                })
            
            print(f'            Estatística Treino {nome}:\n\n{cr_treino}')
            print(f'            Estatística Teste {nome}:\n\n{cr_teste}')
            
            # cmtreino, cmteste = confusion_matrix(y_treino, y_treino_predict), confusion_matrix(y_teste, y_teste_predict)
            # cmplottreino, cmplotteste = ConfusionMatrixDisplay(cmtreino, display_labels=[0,1]), ConfusionMatrixDisplay(cmteste, display_labels=[0,1])

            # fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(10, 8))
            # axes = axes.flatten()

            # cmplottreino.plot(ax=axes[0],cmap="Blues", colorbar=False)
            # axes[0].set_title(f"{nome} Matrix Confusão - Treino Set")
            # cmplotteste.plot(ax=axes[1], cmap="Blues", colorbar=False)
            # axes[1].set_title(f"{nome} Matrix Confusão - Teste Set")

            # plt.figure(figsize=(8,6))
            # plt.title(f"CURVA ROC")
            # plt.plot(roc_treino[0],roc_treino[1], color='red')
            # plt.plot(roc_teste[0],roc_teste[1], color='green')
            # plt.grid(True)
            # plt.plot([0,1],[0,1],color='black',linestyle= '--')
            # plt.legend([f"Treino: {100*auc_treino:.4f}",
            #         f"Teste: {100*auc_teste:.4f}"])

            plt.show()