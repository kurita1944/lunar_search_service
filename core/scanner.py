import os
import rasterio
import numpy as np
from ultralytics import YOLO


class LunarScanner:
    def __init__(self, model_path):
        print(f"Загрузка модели из: {model_path}")
        self.model = YOLO(model_path)
        self.tile_size = 640
        self.overlap = 0.2
        self.stride = int(self.tile_size * (1 - self.overlap))

    def scan_image(self, tiff_path, confidence_threshold=0.25):
        detections = []
        if not os.path.exists(tiff_path):
            raise FileNotFoundError(f"Файл {tiff_path} не найден.")

        try:
            with rasterio.open(tiff_path) as src:
                width = src.width
                height = src.height

                print(f"Сканирование: {width}x{height} пикс.")
                print(f"CRS снимка: {src.crs}")

                for y in range(0, height, self.stride):
                    for x in range(0, width, self.stride):
                        w_tile = min(self.tile_size, width - x)
                        h_tile = min(self.tile_size, height - y)

                        window = rasterio.windows.Window(x, y, w_tile, h_tile)
                        tile_data = src.read(window=window)

                        if tile_data is None or tile_data.size == 0:
                            continue

                        img_tile = np.moveaxis(tile_data, 0, -1)
                        if img_tile.shape[2] == 1:
                            img_tile = np.repeat(img_tile, 3, axis=2)

                        results = self.model(img_tile, conf=confidence_threshold, verbose=False)

                        for r in results:
                            for box_obj in r.boxes:
                                x1, y1, x2, y2 = box_obj.xyxy[0].cpu().numpy()
                                conf = float(box_obj.conf[0].cpu().numpy())
                                cls_id = int(box_obj.cls[0].cpu().numpy())
                                cls_name = self.model.names[cls_id]

                                global_x_center = x + (x1 + x2) / 2
                                global_y_center = y + (y1 + y2) / 2

                                # Прямое извлечение координат из пространственной матрицы снимка
                                try:
                                    # Метод xy напрямую переводит пиксели в пространственные координаты файла
                                    lon, lat = src.xy(global_y_center, global_x_center)
                                except Exception as e:
                                    print(f"Не удалось извлечь гео-координаты: {e}")
                                    lon, lat = global_x_center, global_y_center

                                detections.append({
                                    "class": cls_name,
                                    "conf": round(conf, 3),
                                    "lat": lat,
                                    "lon": lon,
                                    "x_px": int(global_x_center),
                                    "y_px": int(global_y_center)
                                })

            unique_detections = self._filter_duplicates(detections)
            return unique_detections

        except Exception as e:
            print(f"Ошибка при обработке Rasterio: {e}")
            return []

    def _filter_duplicates(self, detections, px_threshold=20):
        if not detections:
            return []
        detections.sort(key=lambda x: x['conf'], reverse=True)
        kept = []
        for current in detections:
            is_duplicate = False
            for exist in kept:
                dist = ((current['x_px'] - exist['x_px']) ** 2 + (current['y_px'] - exist['y_px']) ** 2) ** 0.5
                if dist < px_threshold and current['class'] == exist['class']:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(current)
        return kept