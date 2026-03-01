import os
import torch
import sys

# --- ПАТЧ ДЛЯ PyTorch 2.6 ---
# Перехватываем стандартную функцию torch.load
# и принудительно отключаем проверку weights_only
_original_load = torch.load


def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)


torch.load = safe_load
# ----------------------------

from flask import Flask
from app.routes import register_routes, init_scanner

# ДОБАВЛЕНО: Импорт объекта базы данных
from app.models import db

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Увеличиваем лимит загрузки (снимки тяжелые), например до 500 МБ
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# === ДОБАВЛЕНО: Настройка базы данных ===
# Читаем URI базы данных из переменных окружения (задается в Docker).
# Если переменной нет, создаем локальный файл SQLite (удобно для тестов без Docker)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///lunar_local.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Привязываем базу данных к нашему Flask-приложению
db.init_app(app)
# ========================================

# Инициализация
with app.app_context():
    # ДОБАВЛЕНО: Автоматически создаем все таблицы по моделям, если их еще нет в базе
    db.create_all()

    # Теперь, когда вызовется LunarScanner -> YOLO(),
    # он будет использовать нашу "безопасную" функцию загрузки
    init_scanner(app)
    register_routes(app)

if __name__ == '__main__':
    # Запускаем сервер
    print("Сервис запускается на http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)