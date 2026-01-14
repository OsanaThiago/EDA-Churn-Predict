import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, recall_score, average_precision_score, precision_recall_curve

def get_performance(resultados,X_treino,y_treino,X_teste,y_teste):
   for resultado in resultados:
        
      nome = resultado['nome']
      pipeline = resultado['pipeline']
      id_run = resultado['id_run']
      
      with mlflow.start_run(run_id=id_run):
        y_treino_proba = pipeline.predict_proba(X_treino)[:,1]
        y_teste_proba = pipeline.predict_proba(X_teste)[:,1]

        precision, recall, threshold = precision_recall_curve(y_treino, y_treino_proba)
        precision = precision[:-1]
        recall = recall[:-1]

        target_recall = 0.70
        min_precision = 0.50
        bestthreshold = 0

        for p, r, t in zip(precision, recall, threshold):
            if r >= target_recall and p >= min_precision:
                bestthreshold = t
                break

        if bestthreshold == 0:
            best_f1 = 0
            for p, r, t in zip(precision, recall, threshold):
                if p + r > 0:
                    f1 = 2 * p * r / (p + r)
                    if f1 > best_f1:
                        best_f1 = f1
                        bestthreshold = t

        y_treino_predict = (y_treino_proba >= bestthreshold).astype(int)
        auc_treino = roc_auc_score(y_treino, y_treino_proba)
        roc_treino = roc_curve(y_treino, y_treino_proba)
        cr_treino = classification_report(y_treino, y_treino_predict, target_names=['Não churn','Churn'],zero_division=0)
        avg_treino = average_precision_score(y_treino, y_treino_proba)
        recall_treino = recall_score(y_treino, y_treino_predict)

        y_teste_predict = (y_teste_proba >= bestthreshold).astype(int)
        auc_teste = roc_auc_score(y_teste, y_teste_proba)
        roc_teste = roc_curve(y_teste, y_teste_proba)
        cr_teste = classification_report(y_teste, y_teste_predict, target_names=['Não churn','Churn'],zero_division=0)
        avg_teste = average_precision_score(y_teste, y_teste_proba)
        recall_teste = recall_score(y_teste, y_teste_predict)
        
        mlflow.log_metrics({
            "auc_treino": auc_treino,
            "auc_teste": auc_teste,
            "prc_treino": avg_treino,
            "prc_teste": avg_teste,
            "recall_treino": recall_treino,
            "recall_teste": recall_teste,
            "threshold": bestthreshold
        })

      prprecision_treino, prrecall_treino,_ = precision_recall_curve(y_treino, y_treino_proba)
      prprecision_teste, prrecall_teste,_ = precision_recall_curve(y_teste, y_teste_proba)
      
      print(f'            Estatística Treino {nome}:\n\n{cr_treino}')
      print(f'            Estatística Teste {nome}:\n\n{cr_teste}')
      
      cmtreino, cmteste = confusion_matrix(y_treino, y_treino_predict), confusion_matrix(y_teste, y_teste_predict)
      cmplottreino, cmplotteste = ConfusionMatrixDisplay(cmtreino, display_labels=[0,1]), ConfusionMatrixDisplay(cmteste, display_labels=[0,1])

      fig, axes = plt.subplots(2, 2, figsize=(11, 9))
      axes = axes.flatten()

      cmplottreino.plot(ax=axes[0], cmap="Blues", colorbar=False)
      axes[0].set_title(f"{nome} Matrix Confusão - Treino Set")
      cmplotteste.plot(ax=axes[1], cmap="Blues", colorbar=False)
      axes[1].set_title(f"{nome} Matrix Confusão - Teste Set")

      axes[2].set_title(f"CURVA ROC")
      axes[2].plot(roc_treino[0], roc_treino[1], color='red')
      axes[2].plot(roc_teste[0], roc_teste[1], color='green')
      axes[2].plot([0,1], [0,1], color='black', linestyle= '--')
      axes[2].set_xlabel("1 - Especificidade")
      axes[2].set_ylabel("Sensibilidade")
      axes[2].legend([f"Treino: {100*auc_treino:.4f}", f"Teste: {100*auc_teste:.4f}"], loc='upper left')
      axes[2].grid(True)

      axes[3].set_title(f"Precision-Recall Curve ({nome})")
      axes[3].plot(prrecall_treino, prprecision_treino)
      axes[3].plot(prrecall_teste, prprecision_teste)
      axes[3].set_xlabel("Recall")
      axes[3].set_ylabel("Precision")
      axes[3].legend([f"Treino {avg_treino:.3f}",f"Teste: {avg_teste:.3f}"], loc='upper left')
      axes[3].grid(True)
      
      plt.tight_layout()
      plt.show()