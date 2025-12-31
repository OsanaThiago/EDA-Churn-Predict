import kaggle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def loading_data():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files("radheshyamkollipara/bank-customer-churn", path='./data', unzip=True)
    df = pd.read_csv("data/Customer-Churn-Records.csv")
    df = df.drop(columns=["RowNumber","CustomerId","Surname"])
    return df
    
def preprocessing(df:pd.DataFrame, drop_cols=[]):
    X = df.drop(columns=['Exited'] + drop_cols)
    y = df['Exited']

    cat = X.select_dtypes(object).columns.to_list()
    num = X.select_dtypes([float,int]).columns.tolist()
    X[num] = X[num].astype('float64') #converter int pra float por causa do funcionamento do mlflow 

    X_treino, X_teste, y_treino, y_teste = train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)

    preprocessar = ColumnTransformer(
        transformers = [
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore",sparse_output=False), cat)
        ]
    )

    return X_treino, X_teste, y_treino, y_teste, preprocessar


# print("Taxa variavel resposta geral:", df['Exited'].mean())
# print("Taxa variavel resposta treino:", y_treino.mean())
# print("Taxa variavel resposta teste:", y_teste.mean())
# print(X_treino.isna().sum())


