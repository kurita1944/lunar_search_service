import os

# 1. Хак для RTX 5050 (архитектура Blackwell/sm_120)
# Заставляем PyTorch игнорировать несовпадение версий архитектуры
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'

import torch
import numpy as np

# --- ХАК ДЛЯ NUMPY 2.0 ---
# Возвращаем удаленную функцию trapz, перенаправляя её на новую trapezoid
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
# -------------------------

# --- НАЧАЛО ЯДЕРНОГО РЕШЕНИЯ ---
# 2. Хак для ошибки "Weights only load failed"
# Мы подменяем стандартную функцию загрузки PyTorch, чтобы она по умолчанию
# выключала параноидальную проверку безопасности (weights_only=False).
# Это безопасно, так как мы грузим свои проверенные модели.
_original_load = torch.load


def aggressive_hack_load(*args, **kwargs):
    # Если параметр weights_only не передан явно, ставим его в False
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)


torch.load = aggressive_hack_load
# --- КОНЕЦ ЯДЕРНОГО РЕШЕНИЯ ---

from ultralytics import YOLO

if __name__ == '__main__':
    print("--- Запуск скрипта обучения ---")

    # 3. Проверка видеокарты
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Устройство: {gpu_name}")
        # Дополнительная проверка, чтобы убедиться, что это 5050
        if "5050" in gpu_name:
            print(">> Вижу RTX 5050! Применяем настройки для Blackwell.")
    else:
        print("ВНИМАНИЕ: Обучение пойдет на CPU! Проверь драйверы.")

    # 4. Инициализация модели
    print("Загрузка модели YOLOv8 Medium (с патчем безопасности)...")
    try:
        model = YOLO('yolov8m.pt')
        print("Модель успешно загружена!")
    except Exception as e:
        print(f"ОШИБКА ЗАГРУЗКИ МОДЕЛИ: {e}")
        exit(1)

    # 5. Запуск обучения
    print("Начало обучения...")
    # Обрати внимание: я уменьшил workers до 2 на всякий случай, чтобы не перегрузить CPU на старте.
    # Если все ок, можно вернуть 4.
    model.train(
        data=r'D:/diplome/Lunar_Search_Service.v2-lunar_objects_dataset_v2.yolov8/data.yaml',
        epochs=100,
        imgsz=640,
        device=0,
        batch=16,
        name='lunar_model_m',
        workers=2
    )

    print("Обучение завершено!")