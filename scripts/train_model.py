import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

def train_and_evaluate():
    # 1. Определяем пути
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, 'data')
    models_dir = os.path.join(project_dir, 'models')
    reports_dir = os.path.join(project_dir, 'reports')
    
    # 2. Создаем необходимые папки
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 3. Полный путь к данным
    processed_data_path = os.path.join(data_dir, 'processed_train.csv')
    
    # 4. Проверяем существование файла
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Файл {processed_data_path} не найден! Сначала выполните preprocessing.py")
    
    # 5. Загрузка данных
    df = pd.read_csv(processed_data_path)
    
    # 6. Разделение данных
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 7. Обучение модели
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 8. Оценка
    y_pred = model.predict(X_test)
    metrics = {
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    
    # 9. Сохранение результатов
    joblib.dump(model, os.path.join(models_dir, 'model.pkl'))
    with open(os.path.join(reports_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    # 10. Визуализация
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
# Форматирование осей
    ax = plt.gca()
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.xlabel("Реальные значения ($)", fontsize=12)
    plt.ylabel("Предсказанные значения ($)", fontsize=12)
    plt.title("Сравнение реальных и предсказанных цен", fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()   
    plt.close()
    print("Модель успешно обучена и сохранена!")

if __name__ == '__main__':
    train_and_evaluate()