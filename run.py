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

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Увеличиваем лимит загрузки (снимки тяжелые), например до 500 МБ
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Инициализация
with app.app_context():
    # Теперь, когда вызовется LunarScanner -> YOLO(),
    # он будет использовать нашу "безопасную" функцию загрузки
    init_scanner(app)
    register_routes(app)

if __name__ == '__main__':
    # Запускаем сервер
    print("Сервис запускается на http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)