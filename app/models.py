from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Инициализируем объект базы данных
db = SQLAlchemy()


class Image(db.Model):
    __tablename__ = 'images'

    image_id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    resolution = db.Column(db.Float, nullable=True)  # Разрешение снимка (м/пикс)
    file_path = db.Column(db.Text, nullable=False)

    # Связь "один-ко-многим" с результатами детекции
    detections = db.relationship('Detection', backref='image', lazy=True)


class ObjectClass(db.Model):
    __tablename__ = 'classes'

    class_id = db.Column(db.Integer, primary_key=True)
    class_name = db.Column(db.String(100), nullable=False, unique=True)

    # Связь "один-ко-многим" с результатами детекции
    detections = db.relationship('Detection', backref='object_class', lazy=True)


class Detection(db.Model):
    __tablename__ = 'detections'

    detection_id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey('images.image_id'), nullable=False)
    class_id = db.Column(db.Integer, db.ForeignKey('classes.class_id'), nullable=False)

    bbox_x = db.Column(db.Float, nullable=False)
    bbox_y = db.Column(db.Float, nullable=False)
    bbox_w = db.Column(db.Float, nullable=False)
    bbox_h = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float, nullable=False)