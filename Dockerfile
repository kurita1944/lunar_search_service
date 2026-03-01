# Используем официальный образ Python 3.11, как заявлено в дипломе
FROM python:3.11-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Устанавливаем системные зависимости для OpenCV и GDAL (rasterio)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gdal-bin \
    libgdal-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем переменные окружения для GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt
# Доустанавливаем ORM библиотеку
RUN pip install --no-cache-dir Flask-SQLAlchemy

# Копируем исходный код проекта
COPY . .

# Открываем порт 5000
EXPOSE 5000

# Команда для запуска приложения
CMD ["python", "run.py"]