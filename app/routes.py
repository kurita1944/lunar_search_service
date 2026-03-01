import os
import cv2
from flask import render_template, request, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename
from core.scanner import LunarScanner

# ДОБАВЛЕНО: Импортируем базу данных и наши модели ORM
from app.models import db, Image, ObjectClass, Detection

# Глобальная переменная для сканера
scanner = None


def init_scanner(app):
    global scanner
    # ИСПРАВЛЕНИЕ ЗДЕСЬ: убрали '..'
    # Теперь путь строится от корня проекта прямо в папку data
    model_path = os.path.join(app.root_path, 'data', 'weights', 'best.pt')

    # Добавим проверку для отладки
    print(f"Попытка загрузить модель из: {model_path}")
    if not os.path.exists(model_path):
        print("ОШИБКА: Файл модели не найден по этому пути!")

    scanner = LunarScanner(model_path)


def register_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            return jsonify({'error': 'Нет файла'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400

        if file:
            filename = secure_filename(file.filename)

            # ИСПРАВЛЕНИЕ ЗДЕСЬ: тоже убрали '..'
            # Пути к папкам загрузки и обработки
            upload_folder = os.path.join(current_app.root_path, 'data', 'uploads')
            processed_folder = os.path.join(current_app.root_path, 'data', 'processed')

            os.makedirs(upload_folder, exist_ok=True)
            os.makedirs(processed_folder, exist_ok=True)

            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)

            # --- ЗАПУСК НЕЙРОСЕТИ ---
            try:
                # 1. Сканируем (порог по умолчанию 0.25 из scanner.py)
                detections = scanner.scan_image(filepath)

                # 2. Генерируем чистое превью для браузера
                import rasterio
                import numpy as np

                orig_width = 0
                orig_height = 0

                with rasterio.open(filepath) as src:
                    orig_width = src.width
                    orig_height = src.height

                    factor = 1
                    if src.width > 2000:
                        factor = 0.5

                    out_shape = (src.count, int(src.height * factor), int(src.width * factor))
                    img_data = src.read(out_shape=out_shape)
                    img_vis = np.moveaxis(img_data, 0, -1)

                    if img_vis.dtype == 'uint16':
                        img_vis = (img_vis / 256).astype('uint8')
                    elif img_vis.dtype == 'float32':
                        img_vis = (img_vis * 255).astype('uint8')

                    if len(img_vis.shape) == 2 or img_vis.shape[2] == 1:
                        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
                    else:
                        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

                    # Сохраняем чистую картинку
                    result_filename = f"res_{filename}.jpg"
                    result_path = os.path.join(processed_folder, result_filename)
                    cv2.imwrite(result_path, img_vis)

                # === ДОБАВЛЕНИЕ В БАЗУ ДАННЫХ ===

                # Шаг 1: Записываем информацию о загруженном снимке
                new_image = Image(
                    filename=filename,
                    file_path=filepath,
                    resolution=None  # Можно позже добавить чтение разрешения из метаданных (м/пикс)
                )
                db.session.add(new_image)
                db.session.commit()  # Коммитим, чтобы база присвоила снимку image_id

                # Шаг 2: Проходимся по всем найденным объектам и сохраняем их
                for det in detections:
                    class_name = det.get('class')

                    # Проверяем, существует ли такой класс объектов в базе (АМС, Кратер и т.д.)
                    # Если нет - добавляем его в справочник Classes
                    obj_class = ObjectClass.query.filter_by(class_name=class_name).first()
                    if not obj_class:
                        obj_class = ObjectClass(class_name=class_name)
                        db.session.add(obj_class)
                        db.session.commit()  # Коммитим для получения class_id

                    # Создаем запись результата детекции
                    # Примечание: width и height установлены в 0, так как твой scanner.py
                    # возвращает только центральные координаты объекта (x_px, y_px).
                    new_detection = Detection(
                        image_id=new_image.image_id,
                        class_id=obj_class.class_id,
                        bbox_x=det.get('x_px', 0),
                        bbox_y=det.get('y_px', 0),
                        bbox_w=0,
                        bbox_h=0,
                        confidence=det.get('conf', 0)
                    )
                    db.session.add(new_detection)

                # Сохраняем все детекции одним разом
                db.session.commit()
                # === КОНЕЦ ДОБАВЛЕНИЯ В БАЗУ ДАННЫХ ===

                return jsonify({
                    'status': 'success',
                    'image_url': f'/results/{result_filename}',
                    'detections': detections,
                    'orig_width': orig_width,
                    'orig_height': orig_height,
                    'count': len(detections)
                })

            except Exception as e:
                print(f"Ошибка обработки: {e}")
                db.session.rollback()  # В случае ошибки откатываем изменения в базе
                return jsonify({'error': str(e)}), 500

    @app.route('/results/<filename>')
    def uploaded_file(filename):
        # ИСПРАВЛЕНИЕ ЗДЕСЬ: тоже убрали '..'
        dir_path = os.path.join(current_app.root_path, 'data', 'processed')
        return send_from_directory(dir_path, filename)

    @app.route('/history')
    def history():
        # Запрашиваем все снимки из базы, сортируя по дате (от новых к старым)
        images = Image.query.order_by(Image.upload_date.desc()).all()

        # Словарь для хранения статистики по каждому снимку
        stats = {}

        for img in images:
            class_counts = {}
            conf_0_25 = 0
            conf_25_50 = 0
            conf_50_100 = 0

            for det in img.detections:
                # Подсчет по классам
                c_name = det.object_class.class_name
                class_counts[c_name] = class_counts.get(c_name, 0) + 1

                # Подсчет по уровню уверенности (confidence)
                if det.confidence <= 0.25:
                    conf_0_25 += 1
                elif det.confidence <= 0.50:
                    conf_25_50 += 1
                else:
                    conf_50_100 += 1

            # Сохраняем собранные данные по ID снимка
            stats[img.image_id] = {
                'classes': class_counts,
                'conf_0_25': conf_0_25,
                'conf_25_50': conf_25_50,
                'conf_50_100': conf_50_100
            }

        # Передаем и снимки, и словарь со статистикой в шаблон
        return render_template('history.html', images=images, stats=stats)