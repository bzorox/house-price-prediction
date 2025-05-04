import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def fill_missing(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    categ_cols = df.select_dtypes(include=['object']).columns
    for col in categ_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def encode_categorical(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    return df

def remove_outliers(df, column='SalePrice', threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return df[(z_scores.abs() < threshold)]

def preprocess_data(input_path, output_path):
    # Создаем папку, если ее нет
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Проверяем существование входного файла
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Входной файл {input_path} не найден!")
    
    df = pd.read_csv(input_path)
    df = fill_missing(df)
    df = encode_categorical(df)
    df = remove_outliers(df)
    df.to_csv(output_path, index=False)
    print(f"Данные успешно обработаны и сохранены в {output_path}")
    return df

if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preprocess_data(
        input_path=os.path.join(project_dir, 'data', 'train.csv'),
        output_path=os.path.join(project_dir, 'data', 'processed_train.csv')
    )