# Road Damage Detection (RDD2022)
**YOLOv8 + Albumentations** — детекция дефектов дорожного полотна (RDD2022).  
Ключевая идея: Ultralytics YOLOv8 + кастомные Albumentations → устойчивое обучение и inference в реальном времени.

---

## Суть проекта
Модель детектирует 5 классов повреждений:
`longitudinal crack`, `transverse crack`, `alligator crack`, `other corruption`, `pothole`.

Цели:
- mAP@50: ориентир 50–60%;
- FPS: ориентир ≥ 30.

Результат: итоговый пайплайн на базе **YOLOv8m** с кастомным тренером, адаптированным для Albumentations.

---

## Быстрый старт

1. Установите зависимости:
```bash
pip install ultralytics albumentations opencv-python-headless squarify pandas matplotlib tqdm
```

2. Задайте значения для доступа к Kaggle Api:
```python
KAGGLE_USERNAME=...
KAGGLE_KEY=...
```

3. Запустите блоки: Подготовка, Utils, EDA
4. Пример — запуск короткого прогона (baseline):

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(
    data=f'{MAIN_DIRECTORY}/rdd2022.yaml',
    epochs=7,
    imgsz=640,
    batch=4,
    device=0,
    save=True,
    cache=True,
    project='/content/saves',
    name='Baseline',
    pretrained=True,
    fraction=0.1
)
```

---

## Основные компоненты

* EDA: статистики размеров, плотности bbox, heatmap центров, проверки качества разметки.
* Preprocessing: letterbox (LongestMaxSize + PadIfNeeded) → imgsz=640.
* Augmentations (staged):
  * Stage1: spatial (crop, pad, flip)
  * Stage2: dropout (Coarse/Grid/Constrained)
  * Stage3: color (ToGray, ChannelDropout, Brightness/Contrast, Gamma, Jitter)
  * Stage4: photometric/weather (SunFlare, Shadow)
* CustomTrainer: заменяет Ultralytics Albumentations на кастомные Albumentations.

---

## Эксперименты (ключевые выводы)

* **Оптимизатор:** SGD (momentum) лучше в коротких прогонах, стабильнее по mAP@50.
* **Scheduler:** cosine дает небольшое улучшение.
* **Batch:** 32 - компромисс между точностью и затратами ресурсов.
* **Model size:** yolov8m выбран как оптимум качество/FPS; yolo11m потребляет значительно больше VRAM без явного выигрыша в текущих настройках.
* **Multi-scale:** scale=0.5.

## Оценка / Inference

* Валидация: `model.val(data=MAIN_DIRECTORY / "rdd2022.yaml", imgsz=640, split="test")`
* Визуализация: `show_yolo_predictions(model, img_path, class_names, conf=0.4, show_gt=True)`

## Возможные улучшения / дальнейшая работа

* Доработать и протестировать Mosaic-augmentation совместно с Albumentations (сложная интеграция в Ultralitics pipeline).
* Пробовать другие архитектуры (RT-DETR, кастомные backbones) с отдельным подбором аугментаций.
* Обучение на 50–80 эпох для финальной модели.
