from run import app
from app.models import Image, ObjectClass, Detection

# Нам нужен контекст приложения, чтобы работать с базой
with app.app_context():
    print("=== ПРОВЕРКА БАЗЫ ДАННЫХ ===\n")

    # Проверяем снимки
    images = Image.query.all()
    print(f"Найдено снимков: {len(images)}")
    for img in images:
        print(f" [Снимок ID: {img.image_id}] Файл: {img.filename} | Загружен: {img.upload_date}")

    print("-" * 30)

    # Проверяем классы объектов
    classes = ObjectClass.query.all()
    print(f"Найдено классов: {len(classes)}")
    for c in classes:
        print(f" [Класс ID: {c.class_id}] Название: {c.class_name}")

    print("-" * 30)

    # Проверяем детекции
    detections = Detection.query.all()
    print(f"Найдено детекций: {len(detections)}")
    for d in detections:
        print(f" [Детекция ID: {d.detection_id}] Снимок: {d.image_id} | Класс: {d.class_id} | Уверенность: {d.confidence:.2f} | Координаты: X={d.bbox_x}, Y={d.bbox_y}")