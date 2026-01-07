import kaggle
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def loading_data():
    DATA_DIR = (Path.cwd().parent / "data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files("radheshyamkollipara/bank-customer-churn", path=DATA_DIR, unzip=True)
    df = pd.read_csv(DATA_DIR/'Customer-Churn-Records.csv')
    df = df.drop(columns=["RowNumber","CustomerId","Surname",'Complain'])

    return df
    
def preprocessing(df:pd.DataFrame):
    X = df.drop(columns=['Exited'])
    y = df['Exited']

    cat = X.select_dtypes(object).columns.to_list()
    num = X.select_dtypes([float,int]).columns.tolist()
    X[num] = X[num].astype('float64') #converter int pra float por causa do funcionamento do mlflow 

    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

    preprocessar = ColumnTransformer(
        transformers = [
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
        ]
    )

    return X_treino, X_teste, y_treino, y_teste, preprocessar






