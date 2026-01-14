import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    """Пользовательский датасет для обучения"""
    def __init__(self, image_paths, labels, class_to_idx, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label_idx = self.class_to_idx[self.labels[idx]]
        return image, label_idx

class ImageClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Классификатор изображений")
        self.root.geometry("1400x800")
        self.root.configure(bg="#f0f0f1")

        # Инициализация переменных
        self.model = None
        self.model_loaded = False
        self.model_arch = "efficientnet_b0"
        self.current_image = None
        self.current_image_path = None
        self.checkpoint_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {self.device}")

        # Данные для обучения
        self.dataset_folder = "custom_dataset"
        self.train_folder = os.path.join(self.dataset_folder, "train")
        self.test_folder = os.path.join(self.dataset_folder, "test")
        self.checkpoints_folder = "checkpoints"

        # Создание папок
        self.create_folders()

        # Классы и метки
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Трансформации
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Переменные для обучения
        self.epochs = 10
        self.batch_size = 16
        self.learning_rate = 0.001
        self.is_training = False
        self.training_thread = None

        # Статистика обучения
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

        # Создание интерфейса
        self.create_widgets()

        # Загрузка последнего чекпоинта если есть
        self.load_latest_checkpoint()

    def create_folders(self):
        """Создание необходимых папок"""
        folders = [self.dataset_folder, self.train_folder,
                   self.test_folder, self.checkpoints_folder]

        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"Создана папка: {folder}")

    def create_widgets(self):
        """Создание графического интерфейса"""
        # Главный контейнер
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Заголовок
        header_frame = tk.Frame(main_frame, bg="#2c3e50")
        header_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(header_frame, text="Классификатор изображений",
                 font=("Arial", 20, "bold"),
                 fg="white", bg="#2c3e50").pack(pady=15)

        # Панель с вкладками
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Вкладка 1: Управление данными
        data_tab = ttk.Frame(notebook)
        notebook.add(data_tab, text="📁 Данные")

        # Вкладка 2: Обучение
        training_tab = ttk.Frame(notebook)
        notebook.add(training_tab, text="🎓 Обучение")

        # Вкладка 3: Классификация
        classify_tab = ttk.Frame(notebook)
        notebook.add(classify_tab, text="🔍 Классификация")

        # Создание содержимого вкладок
        self.create_data_tab(data_tab)
        self.create_training_tab(training_tab)
        self.create_classify_tab(classify_tab)

        # Статус бар
        self.status_bar = ttk.Label(main_frame, text="Готов к работе",
                                    relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, pady=(10, 0))

    def create_data_tab(self, parent):
        """Создание вкладки управления данными"""
        # Левая панель - загрузка данных
        left_panel = ttk.LabelFrame(parent, text="Загрузка данных", padding="15")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Правая панель - управление данными
        right_panel = ttk.LabelFrame(parent, text="Управление данными", padding="15")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Левая панель: Загрузка данных
        tk.Label(left_panel, text="1. Создать новый класс:",
                 font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))

        class_frame = ttk.Frame(left_panel)
        class_frame.pack(fill=tk.X, pady=(0, 15))

        self.class_name_var = tk.StringVar()
        ttk.Entry(class_frame, textvariable=self.class_name_var,
                  font=("Arial", 10), width=20).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(class_frame, text="Создать класс",
                   command=self.create_class).pack(side=tk.LEFT)

        tk.Label(left_panel, text="2. Загрузить изображения в класс:",
                 font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))

        # Выбор класса
        self.class_var = tk.StringVar()
        self.class_combobox = ttk.Combobox(left_panel, textvariable=self.class_var,
                                           state="readonly", font=("Arial", 10))
        self.class_combobox.pack(fill=tk.X, pady=(0, 10))

        # Кнопка загрузки изображений
        ttk.Button(left_panel, text="📁 Загрузить изображения в выбранный класс",
                   command=self.load_images_to_class).pack(fill=tk.X, pady=(0, 15))

        # Автоматическое разделение
        tk.Label(left_panel, text="3. Разделить данные на train/test:",
                 font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))

        split_frame = ttk.Frame(left_panel)
        split_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(split_frame, text="Test size (%):").pack(side=tk.LEFT, padx=(0, 10))
        self.test_size_var = tk.DoubleVar(value=20.0)
        ttk.Spinbox(split_frame, from_=10, to=40, increment=5,
                    textvariable=self.test_size_var, width=10).pack(side=tk.LEFT)

        ttk.Button(split_frame, text="Разделить",
                   command=self.split_dataset).pack(side=tk.LEFT, padx=(10, 0))

        # Просмотр изображений
        tk.Label(left_panel, text="4. Просмотр изображений класса:",
                 font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))

        self.preview_class_var = tk.StringVar()
        self.preview_combobox = ttk.Combobox(left_panel,
                                             textvariable=self.preview_class_var,
                                             state="readonly", font=("Arial", 10))
        self.preview_combobox.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(left_panel, text="👁️ Показать изображения",
                   command=self.preview_class_images).pack(fill=tk.X)

        # Правая панель: Информация о данных
        # Статистика
        self.data_stats_text = scrolledtext.ScrolledText(right_panel, height=15,
                                                         font=("Consolas", 9),
                                                         bg="#fafafa")
        self.data_stats_text.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # Управление классами
        class_manage_frame = ttk.LabelFrame(right_panel, text="Управление классами", padding="10")
        class_manage_frame.pack(fill=tk.X)

        btn_frame = ttk.Frame(class_manage_frame)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="🔄 Обновить список классов",
                   command=self.update_classes_list).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(btn_frame, text="🗑️ Удалить выбранный класс",
                   command=self.delete_class).pack(side=tk.LEFT)

        # Обновляем список классов
        self.update_classes_list()

    def create_training_tab(self, parent):
        """Создание вкладки обучения"""
        # Левая панель - настройки обучения
        left_panel = ttk.LabelFrame(parent, text="Настройки обучения", padding="15")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Правая панель - управление обучением
        right_panel = ttk.LabelFrame(parent, text="Управление обучением", padding="15")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Левая панель: Параметры
        tk.Label(left_panel, text="Архитектура модели: EfficientNet-B0",
                 font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))

        # Настройки обучения
        tk.Label(left_panel, text="Гиперпараметры:",
                 font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))

        # Эпохи
        epoch_frame = ttk.Frame(left_panel)
        epoch_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(epoch_frame, text="Количество эпох:").pack(side=tk.LEFT, padx=(0, 10))
        self.epochs_var = tk.IntVar(value=10)
        ttk.Spinbox(epoch_frame, from_=1, to=100, textvariable=self.epochs_var,
                    width=10).pack(side=tk.LEFT)

        # Batch size
        batch_frame = ttk.Frame(left_panel)
        batch_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(batch_frame, text="Batch size:").pack(side=tk.LEFT, padx=(0, 10))
        self.batch_size_var = tk.IntVar(value=16)
        ttk.Spinbox(batch_frame, from_=1, to=64, textvariable=self.batch_size_var,
                    width=10).pack(side=tk.LEFT)

        # Learning rate
        lr_frame = ttk.Frame(left_panel)
        lr_frame.pack(fill=tk.X, pady=(0, 15))
        tk.Label(lr_frame, text="Learning rate:").pack(side=tk.LEFT, padx=(0, 10))
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Spinbox(lr_frame, from_=0.0001, to=0.01, increment=0.0001,
                    textvariable=self.lr_var, width=10,
                    format="%.4f").pack(side=tk.LEFT)

        # Информация об оптимизаторе (Adam - используется всегда)
        opt_frame = ttk.Frame(left_panel)
        opt_frame.pack(fill=tk.X, pady=(0, 15))
        tk.Label(opt_frame, text="Оптимизатор:").pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(opt_frame, text="Adam",
                 font=("Arial", 9), foreground="#3498db").pack(side=tk.LEFT)

        # Правая панель: Управление обучением
        # Кнопки управления
        self.btn_train = tk.Button(right_panel, text="🎓 Начать обучение",
                                   command=self.start_training,
                                   font=("Arial", 12, "bold"),
                                   bg="#27ae60", fg="white",
                                   padx=20, pady=10,
                                   cursor="hand2")
        self.btn_train.pack(fill=tk.X, pady=(0, 10))

        self.btn_stop = tk.Button(right_panel, text="⏹️ Остановить обучение",
                                  command=self.stop_training,
                                  font=("Arial", 10),
                                  bg="#e74c3c", fg="white",
                                  padx=20, pady=8,
                                  state=tk.DISABLED,
                                  cursor="hand2")
        self.btn_stop.pack(fill=tk.X, pady=(0, 15))

        # Загрузка чекпоинта
        ttk.Label(right_panel, text="Загрузка чекпоинта:",
                  font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))

        checkpoint_frame = ttk.Frame(right_panel)
        checkpoint_frame.pack(fill=tk.X, pady=(0, 15))

        self.btn_load_checkpoint = ttk.Button(checkpoint_frame, text="Загрузить чекпоинт",
                                              command=self.load_training_checkpoint)
        self.btn_load_checkpoint.pack(side=tk.LEFT, padx=(0, 10))

        self.checkpoint_label = ttk.Label(checkpoint_frame, text="Нет чекпоинта",
                                          foreground="gray")
        self.checkpoint_label.pack(side=tk.LEFT)

        # Информация о модели
        model_info_frame = ttk.LabelFrame(right_panel, text="Информация о модели", padding="10")
        model_info_frame.pack(fill=tk.BOTH, expand=True)

        self.model_info_text = scrolledtext.ScrolledText(model_info_frame, height=10,
                                                         font=("Consolas", 9),
                                                         bg="#fafafa")
        self.model_info_text.pack(fill=tk.BOTH, expand=True)

    def create_classify_tab(self, parent):
        """Создание вкладки классификации"""
        # Левая панель - загрузка изображения
        left_panel = ttk.LabelFrame(parent, text="Классификация", padding="15")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Правая панель - результаты
        right_panel = ttk.LabelFrame(parent, text="Результаты", padding="15")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Левая панель
        self.btn_load_classify = tk.Button(left_panel, text="📁 Загрузить изображение",
                                           command=self.load_image_for_classification,
                                           font=("Arial", 10, "bold"),
                                           bg="#3498db", fg="white",
                                           padx=20, pady=10,
                                           cursor="hand2")
        self.btn_load_classify.pack(fill=tk.X, pady=(0, 15))

        # Отображение изображения
        self.classify_image_label = tk.Label(left_panel, text="Изображение не загружено",
                                             bg="white", relief=tk.SUNKEN,
                                             font=("Arial", 10))
        self.classify_image_label.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # Кнопка классификации
        self.btn_classify = tk.Button(left_panel, text="🔍 Классифицировать",
                                      command=self.classify_image,
                                      font=("Arial", 12, "bold"),
                                      bg="#9b59b6", fg="white",
                                      padx=20, pady=12,
                                      state=tk.DISABLED,
                                      cursor="hand2")
        self.btn_classify.pack(fill=tk.X)

        # Правая панель: Результаты
        # Вкладки для результатов
        results_notebook = ttk.Notebook(right_panel)
        results_notebook.pack(fill=tk.BOTH, expand=True)

        # Вкладка 1: Предсказания
        predictions_frame = ttk.Frame(results_notebook)
        results_notebook.add(predictions_frame, text="Предсказания")

        self.predictions_text = scrolledtext.ScrolledText(predictions_frame,
                                                          font=("Consolas", 10),
                                                          bg="#fafafa")
        self.predictions_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Вкладка 2: График уверенности
        confidence_frame = ttk.Frame(results_notebook)
        results_notebook.add(confidence_frame, text="График уверенности")

        self.confidence_figure, self.confidence_ax = plt.subplots(figsize=(6, 4), dpi=80)
        self.confidence_figure.patch.set_facecolor('#f0f0f1')
        self.confidence_canvas = FigureCanvasTkAgg(self.confidence_figure, confidence_frame)
        self.confidence_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def build_model(self, num_classes):
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Заморозка
        for p in model.parameters():
            p.requires_grad = False

        # Классификатор
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

        # Разморозка
        for p in model.classifier.parameters():
            p.requires_grad = True
        for p in model.features[-2:].parameters():
            p.requires_grad = True

        return model.to(self.device)

    def create_class(self):
        """Создание нового класса"""
        class_name = self.class_name_var.get().strip()

        if not class_name:
            messagebox.showwarning("Предупреждение", "Введите название класса")
            return

        # Создаем папку для класса
        class_path = os.path.join(self.dataset_folder, class_name)
        if os.path.exists(class_path):
            messagebox.showinfo("Информация", f"Класс '{class_name}' уже существует")
            return

        os.makedirs(class_path)
        self.update_status(f"Создан новый класс: {class_name}")
        self.update_classes_list()
        self.class_name_var.set("")

    def update_classes_list(self):
        """Обновление списка классов"""
        # Получаем все папки в dataset_folder (исключаем train и test)
        all_items = os.listdir(self.dataset_folder)
        self.classes = []

        for item in all_items:
            item_path = os.path.join(self.dataset_folder, item)
            if os.path.isdir(item_path) and item not in ['train', 'test', 'checkpoints']:
                self.classes.append(item)

        # Обновляем combobox
        self.class_combobox['values'] = self.classes
        self.preview_combobox['values'] = self.classes

        if self.classes:
            self.class_var.set(self.classes[0])
            self.preview_class_var.set(self.classes[0])

        # Обновляем статистику
        self.update_data_stats()

    def load_images_to_class(self):
        """Загрузка изображений в выбранный класс"""
        class_name = self.class_var.get()

        if not class_name:
            messagebox.showwarning("Предупреждение", "Выберите класс")
            return

        class_path = os.path.join(self.dataset_folder, class_name)

        # Выбор файлов
        filetypes = [
            ("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("Все файлы", "*.*")
        ]

        filenames = filedialog.askopenfilenames(
            title=f"Выберите изображения для класса '{class_name}'",
            filetypes=filetypes
        )

        if not filenames:
            return

        # Копируем файлы в папку класса
        copied_count = 0
        for filename in filenames:
            try:
                basename = os.path.basename(filename)
                dest_path = os.path.join(class_path, basename)

                # Если файл с таким именем уже существует, добавляем суффикс
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(basename)
                    dest_path = os.path.join(class_path, f"{name}_{counter}{ext}")
                    counter += 1

                shutil.copy2(filename, dest_path)
                copied_count += 1

            except Exception as e:
                print(f"Ошибка копирования {filename}: {e}")

        self.update_status(f"Загружено {copied_count} изображений в класс '{class_name}'")
        self.update_data_stats()

    def split_dataset(self):
        """Разделение данных на train/test"""
        if not self.classes:
            messagebox.showwarning("Предупреждение", "Нет классов для разделения")
            return

        # Очищаем папки train и test
        for folder in [self.train_folder, self.test_folder]:
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)

        # Создаем структуру папок
        for class_name in self.classes:
            train_class_path = os.path.join(self.train_folder, class_name)
            test_class_path = os.path.join(self.test_folder, class_name)
            os.makedirs(train_class_path, exist_ok=True)
            os.makedirs(test_class_path, exist_ok=True)

            # Получаем все изображения класса
            class_path = os.path.join(self.dataset_folder, class_name)
            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

            if len(images) < 2:
                print(f"Пропускаем класс '{class_name}': слишком мало изображений ({len(images)})")
                continue

            # Разделяем изображения
            test_size = self.test_size_var.get() / 100.0
            train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=42)

            # Копируем в соответствующие папки
            for img in train_imgs:
                src = os.path.join(class_path, img)
                dst = os.path.join(train_class_path, img)
                shutil.copy2(src, dst)

            for img in test_imgs:
                src = os.path.join(class_path, img)
                dst = os.path.join(test_class_path, img)
                shutil.copy2(src, dst)

            print(f"Класс '{class_name}': train={len(train_imgs)}, test={len(test_imgs)}")

        self.update_status(f"Данные разделены на train/test (test size: {self.test_size_var.get()}%)")
        self.update_data_stats()

    def update_data_stats(self):
        """Обновление статистики данных"""
        self.data_stats_text.delete(1.0, tk.END)

        text = "📊 СТАТИСТИКА ДАННЫХ\n"
        text += "═" * 40 + "\n\n"

        text += f"Количество классов: {len(self.classes)}\n"
        text += "─" * 40 + "\n\n"

        total_train = 0
        total_test = 0

        for class_name in self.classes:
            # Изображения в основном классе
            class_path = os.path.join(self.dataset_folder, class_name)
            class_images = []
            if os.path.exists(class_path):
                class_images = [f for f in os.listdir(class_path)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

            # Изображения в train
            train_class_path = os.path.join(self.train_folder, class_name)
            train_images = []
            if os.path.exists(train_class_path):
                train_images = [f for f in os.listdir(train_class_path)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

            # Изображения в test
            test_class_path = os.path.join(self.test_folder, class_name)
            test_images = []
            if os.path.exists(test_class_path):
                test_images = [f for f in os.listdir(test_class_path)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

            total_train += len(train_images)
            total_test += len(test_images)

            text += f"📁 {class_name}:\n"
            text += f"   Всего: {len(class_images)} изображений\n"
            text += f"   Train: {len(train_images)} изображений\n"
            text += f"   Test:  {len(test_images)} изображений\n\n"

        text += "═" * 40 + "\n"
        text += f"ИТОГО:\n"
        text += f"   Train: {total_train} изображений\n"
        text += f"   Test:  {total_test} изображений\n"
        text += f"   Всего: {total_train + total_test} изображений\n"

        self.data_stats_text.insert(1.0, text)

    def preview_class_images(self):
        """Просмотр изображений класса"""
        class_name = self.preview_class_var.get()

        if not class_name:
            return

        class_path = os.path.join(self.dataset_folder, class_name)
        if not os.path.exists(class_path):
            return

        # Создаем окно для просмотра
        preview_window = tk.Toplevel(self.root)
        preview_window.title(f"Изображения класса: {class_name}")
        preview_window.geometry("800x600")

        # Получаем изображения
        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

        if not images:
            tk.Label(preview_window, text="Нет изображений в этом классе",
                     font=("Arial", 12)).pack(pady=50)
            return

        # Canvas для прокрутки
        canvas = tk.Canvas(preview_window)
        scrollbar = ttk.Scrollbar(preview_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Отображаем изображения
        for i, img_name in enumerate(images[:20]):  # Ограничиваем 20 изображениями
            try:
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path)
                img.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(img)

                frame = ttk.Frame(scrollable_frame)
                frame.pack(pady=5, padx=10, fill=tk.X)

                label_img = tk.Label(frame, image=photo)
                label_img.image = photo
                label_img.pack(side=tk.LEFT, padx=(0, 10))

                label_text = tk.Label(frame, text=f"{img_name}\n{img.size[0]}x{img.size[1]}",
                                      font=("Arial", 9))
                label_text.pack(side=tk.LEFT)

            except Exception as e:
                print(f"Ошибка загрузки {img_name}: {e}")

        if len(images) > 20:
            tk.Label(scrollable_frame,
                     text=f"... и еще {len(images) - 20} изображений",
                     font=("Arial", 10)).pack(pady=10)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def delete_class(self):
        """Удаление выбранного класса"""
        class_name = self.class_var.get()

        if not class_name:
            messagebox.showwarning("Предупреждение", "Выберите класс для удаления")
            return

        if not messagebox.askyesno("Подтверждение",
                                   f"Удалить класс '{class_name}' и все его изображения?"):
            return

        # Удаляем папку класса
        class_path = os.path.join(self.dataset_folder, class_name)
        if os.path.exists(class_path):
            shutil.rmtree(class_path)

        # Удаляем из train и test
        train_class_path = os.path.join(self.train_folder, class_name)
        if os.path.exists(train_class_path):
            shutil.rmtree(train_class_path)

        test_class_path = os.path.join(self.test_folder, class_name)
        if os.path.exists(test_class_path):
            shutil.rmtree(test_class_path)

        self.update_status(f"Удален класс: {class_name}")
        self.update_classes_list()

    def start_training(self):
        """Запуск обучения модели"""
        if not self.classes:
            messagebox.showwarning("Предупреждение", "Сначала создайте классы и загрузите данные")
            return

        # Проверяем, есть ли данные для обучения
        train_images = []
        for class_name in self.classes:
            train_class_path = os.path.join(self.train_folder, class_name)
            if os.path.exists(train_class_path):
                images = [f for f in os.listdir(train_class_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                if images:
                    train_images.extend(images)

        if not train_images:
            messagebox.showwarning("Предупреждение", "Нет данных для обучения. Запустите разделение данных.")
            return

        # Обновляем параметры
        self.epochs = self.epochs_var.get()
        self.batch_size = self.batch_size_var.get()
        self.learning_rate = self.lr_var.get()

        # Запускаем обучение в отдельном потоке
        self.is_training = True
        self.btn_train.config(state=tk.DISABLED, bg="#95a5a6")
        self.btn_stop.config(state=tk.NORMAL, bg="#e74c3c")

        self.training_thread = threading.Thread(target=self.train_model, daemon=True)
        self.training_thread.start()

    def train_model(self):
        """Основная функция обучения модели"""
        try:
            # Подготавливаем данные
            train_image_paths = []
            train_labels = []
            val_image_paths = []
            val_labels = []

            # Создаем mapping классов
            self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}
            self.idx_to_class = {i: class_name for i, class_name in enumerate(self.classes)}

            # Загружаем train данные
            for class_name in self.classes:
                class_path = os.path.join(self.train_folder, class_name)
                if os.path.exists(class_path):
                    images = [f for f in os.listdir(class_path)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                    for img in images:
                        train_image_paths.append(os.path.join(class_path, img))
                        train_labels.append(class_name)

            # Загружаем validation данные
            for class_name in self.classes:
                class_path = os.path.join(self.test_folder, class_name)
                if os.path.exists(class_path):
                    images = [f for f in os.listdir(class_path)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                    for img in images:
                        val_image_paths.append(os.path.join(class_path, img))
                        val_labels.append(class_name)

            if not train_image_paths:
                self.root.after(0, lambda: messagebox.showerror("Ошибка", "Нет данных для обучения"))
                return

            # Создаем датасеты
            train_dataset = CustomDataset(
                train_image_paths,
                train_labels,
                self.class_to_idx,
                self.train_transform
            )

            val_dataset = CustomDataset(
                val_image_paths,
                val_labels,
                self.class_to_idx,
                self.val_transform
            )

            # Создаем DataLoader
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                    shuffle=False, num_workers=0)

            # Инициализируем модель EfficientNet_B0
            num_classes = len(self.classes)
            self.model = self.build_model(num_classes)

            # Оптимизатор Adam и функция потерь
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.learning_rate
            )

            # Обучение
            self.train_loss_history = []
            self.train_acc_history = []
            self.val_loss_history = []
            self.val_acc_history = []

            for epoch in range(self.epochs):
                if not self.is_training:
                    break

                # Режим обучения
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    if not self.is_training:
                        break

                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()

                    # Обновляем прогресс в GUI
                    progress = (epoch * len(train_loader) + batch_idx + 1) / (self.epochs * len(train_loader))
                    self.root.after(0, self.update_training_progress, epoch + 1,
                                    batch_idx + 1, len(train_loader), progress * 100)

                # Валидация
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()

                # Сохраняем метрики
                train_loss_avg = train_loss / len(train_loader)
                train_acc = 100. * train_correct / train_total
                val_loss_avg = val_loss / len(val_loader)
                val_acc = 100. * val_correct / val_total

                self.train_loss_history.append(train_loss_avg)
                self.train_acc_history.append(train_acc)
                self.val_loss_history.append(val_loss_avg)
                self.val_acc_history.append(val_acc)

                # Обновляем GUI
                self.root.after(0, self.update_training_stats, epoch + 1,
                                train_loss_avg, train_acc, val_loss_avg, val_acc)

                # Сохраняем чекпоинт
                if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                    self.save_checkpoint(epoch + 1, train_acc, val_acc)

            # Завершение обучения
            self.model_loaded = True
            self.root.after(0, self.training_completed)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка обучения", str(e)))
            self.root.after(0, self.training_stopped)

    def stop_training(self):
        """Остановка обучения"""
        self.is_training = False
        self.update_status("Обучение остановлено пользователем")

    def training_completed(self):
        """Завершение обучения"""
        self.is_training = False
        self.btn_train.config(state=tk.NORMAL, bg="#27ae60")
        self.btn_stop.config(state=tk.DISABLED, bg="#e74c3c")

        # Обновляем информацию о модели
        self.update_model_info()

        self.update_status("Обучение завершено успешно!")
        messagebox.showinfo("Обучение завершено", "Модель EfficientNet B0 успешно обучена на ваших данных!")

    def training_stopped(self):
        """Обработка остановки обучения"""
        self.is_training = False
        self.btn_train.config(state=tk.NORMAL, bg="#27ae60")
        self.btn_stop.config(state=tk.DISABLED, bg="#e74c3c")

    def update_training_progress(self, epoch, batch, total_batches, progress):
        """Обновление прогресса обучения в GUI"""
        self.status_bar.config(
            text=f"[{datetime.now().strftime('%H:%M:%S')}] Эпоха {epoch}/{self.epochs}, "
                 f"Батч {batch}/{total_batches}, Прогресс: {progress:.1f}%"
        )

    def update_training_stats(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Обновление статистики обучения"""
        # Обновляем текстовую статистику
        self.model_info_text.delete(1.0, tk.END)

        text = "📈 СТАТИСТИКА ОБУЧЕНИЯ\n"
        text += "═" * 50 + "\n\n"

        text += f"Эпоха: {epoch}/{self.epochs}\n"
        text += f"Train Loss: {train_loss:.4f}\n"
        text += f"Train Accuracy: {train_acc:.2f}%\n"
        text += f"Val Loss: {val_loss:.4f}\n"
        text += f"Val Accuracy: {val_acc:.2f}%\n\n"

        if len(self.train_loss_history) > 1:
            text += "📊 ИСТОРИЯ ОБУЧЕНИЯ:\n"
            text += "-" * 30 + "\n"
            for i in range(len(self.train_loss_history)):
                text += f"Эпоха {i + 1}: Train Acc={self.train_acc_history[i]:.2f}%, "
                text += f"Val Acc={self.val_acc_history[i]:.2f}%\n"

        self.model_info_text.insert(1.0, text)

    def save_checkpoint(self, epoch, train_acc, val_acc):
        """Сохранение чекпоинта модели"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"efficientnet_b0_epoch_{epoch}_{timestamp}.pth"
        checkpoint_path = os.path.join(self.checkpoints_folder, checkpoint_name)

        checkpoint = {
            "arch": self.model_arch,
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "classes": self.classes,
            "class_to_idx": self.class_to_idx,
            "idx_to_class": self.idx_to_class,
            "train_loss_history": self.train_loss_history,
            "train_acc_history": self.train_acc_history,
            "val_loss_history": self.val_loss_history,
            "val_acc_history": self.val_acc_history
        }

        torch.save(checkpoint, checkpoint_path)
        self.update_status(f"Сохранен чекпоинт: {checkpoint_name}")

    def load_latest_checkpoint(self):
        """Загрузка последнего чекпоинта"""
        if not os.path.exists(self.checkpoints_folder):
            return

        checkpoints = [f for f in os.listdir(self.checkpoints_folder)
                       if f.endswith('.pth') and 'efficientnet' in f]

        if not checkpoints:
            return

        # Берем самый новый чекпоинт
        latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(
            os.path.join(self.checkpoints_folder, x)))
        checkpoint_path = os.path.join(self.checkpoints_folder, latest_checkpoint)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Восстанавливаем классы
            self.classes = checkpoint['classes']
            self.class_to_idx = checkpoint['class_to_idx']
            self.idx_to_class = checkpoint['idx_to_class']

            # Восстанавливаем историю обучения
            self.train_loss_history = checkpoint.get('train_loss_history', [])
            self.train_acc_history = checkpoint.get('train_acc_history', [])
            self.val_loss_history = checkpoint.get('val_loss_history', [])
            self.val_acc_history = checkpoint.get('val_acc_history', [])

            # Обновляем интерфейс
            self.update_classes_list()

            self.checkpoint_label.config(text=latest_checkpoint, foreground="green")
            self.checkpoint_path = checkpoint_path

            self.update_status(f"Загружен последний чекпоинт: {latest_checkpoint}")

        except Exception as e:
            print(f"Ошибка загрузки чекпоинта: {e}")

    def load_training_checkpoint(self):
        """Загрузка выбранного чекпоинта"""
        filetypes = [
            ("PyTorch модели", "*.pth"),
            ("Все файлы", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Выберите файл чекпоинта",
            filetypes=filetypes,
            initialdir=self.checkpoints_folder
        )

        if not filename:
            return

        try:
            checkpoint = torch.load(filename, map_location=self.device)

            # Восстанавливаем классы
            self.classes = checkpoint['classes']
            self.class_to_idx = checkpoint['class_to_idx']
            self.idx_to_class = checkpoint['idx_to_class']

            # Восстанавливаем историю обучения
            self.train_loss_history = checkpoint.get('train_loss_history', [])
            self.train_acc_history = checkpoint.get('train_acc_history', [])
            self.val_loss_history = checkpoint.get('val_loss_history', [])
            self.val_acc_history = checkpoint.get('val_acc_history', [])

            # Загружаем модель efficientnet_b0
            num_classes = len(self.classes)
            self.model = self.build_model(num_classes)

            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()

            # Обновляем интерфейс
            self.update_classes_list()
            self.update_model_info()

            filename_short = os.path.basename(filename)
            self.checkpoint_label.config(text=filename_short, foreground="green")
            self.checkpoint_path = filename

            self.model_loaded = True

            self.update_status(f"Загружен чекпоинт: {filename_short}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить чекпоинт:\n{str(e)}")

    def update_model_info(self):
        """Обновление информации о модели"""
        self.model_info_text.delete(1.0, tk.END)

        text = "🧠 ИНФОРМАЦИЯ О МОДЕЛИ\n"
        text += "═" * 40 + "\n\n"

        if self.model is None:
            text += "Модель не загружена\n"
            self.model_info_text.insert(1.0, text)
            return

        text += f"Архитектура: EfficientNet-B0\n"
        text += f"Количество классов: {len(self.classes)}\n"
        text += f"Устройство: {self.device}\n"
        text += f"Загружена: {'Да' if self.model_loaded else 'Нет'}\n"
        text += f"Оптимизатор: Adam\n\n"

        # Параметры
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        text += f"Всего параметров: {total_params:,}\n"
        text += f"Обучаемых параметров: {trainable_params:,}\n"
        text += f"Замороженных параметров: {total_params - trainable_params:,}\n\n"

        # Слои
        text += "ОСНОВНЫЕ СЛОИ EfficientNet-B0:\n"
        text += "- features (MBConv блоки)\n"
        text += f"- classifier: {self.model.classifier[1].in_features} → {self.model.classifier[1].out_features}\n"

        self.model_info_text.insert(1.0, text)

    def load_image_for_classification(self):
        """Загрузка изображения для классификации"""
        filetypes = [
            ("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("Все файлы", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Выберите изображение для классификации",
            filetypes=filetypes
        )

        if not filename:
            return

        try:
            # Загрузка и отображение изображения
            image = Image.open(filename)
            self.current_image = image.copy()
            self.current_image_path = filename

            # Масштабирование для отображения
            display_size = (300, 300)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(image)
            self.classify_image_label.config(image=photo, text="")
            self.classify_image_label.image = photo

            # Активация кнопки классификации
            self.btn_classify.config(state=tk.NORMAL, bg="#9b59b6")

            self.update_status(f"Загружено изображение для классификации")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {str(e)}")

    def classify_image(self):
        """Классификация изображения"""
        if not self.model_loaded or self.current_image is None:
            messagebox.showwarning("Предупреждение",
                                   "Сначала загрузите модель и изображение")
            return

        try:
            # Показываем индикатор загрузки
            self.btn_classify.config(text="⏳ Обработка...", state=tk.DISABLED)
            self.root.update()

            # Преобразование изображения
            input_image = self.current_image.convert('RGB')
            input_tensor = self.val_transform(input_image).unsqueeze(0).to(self.device)

            # Классификация
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            # Получение предсказаний
            top3_prob, top3_catid = torch.topk(probabilities, min(3, len(self.classes)))

            # Отображение результатов
            self.display_classification_results(top3_prob, top3_catid)
            self.plot_classification_confidence(top3_prob, top3_catid)

            # Восстанавливаем кнопку
            self.btn_classify.config(text="🔍 Классифицировать", state=tk.NORMAL)

            # Обновляем статус
            best_class_idx = top3_catid[0].item()
            best_class = self.idx_to_class.get(best_class_idx, f"Класс {best_class_idx}")
            confidence = top3_prob[0].item() * 100
            self.update_status(f"Результат: {best_class} ({confidence:.1f}%)")

        except Exception as e:
            # Восстанавливаем кнопку
            self.btn_classify.config(text="🔍 Классифицировать", state=tk.NORMAL)

            messagebox.showerror("Ошибка", f"Ошибка классификации: {str(e)}")

    def display_classification_results(self, probabilities, categories):
        """Отображение результатов классификации"""
        self.predictions_text.delete(1.0, tk.END)

        text = "🏆 РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ\n"
        text += "═" * 40 + "\n\n"

        for i, (prob, cat_idx) in enumerate(zip(probabilities, categories)):
            # Получаем название класса
            class_name = self.idx_to_class.get(cat_idx.item(), f"Класс {cat_idx.item()}")

            # Процент уверенности
            percentage = prob.item() * 100

            # Иконка уверенности
            if percentage > 80:
                confidence_icon = "🟢"
            elif percentage > 50:
                confidence_icon = "🟡"
            elif percentage > 20:
                confidence_icon = "🟠"
            else:
                confidence_icon = "🔴"

            # Форматируем вывод
            text += f"{i + 1}. {class_name}\n"
            text += f"   {confidence_icon} Уверенность: {percentage:.2f}%\n"

            if i < len(probabilities) - 1:
                text += "-" * 30 + "\n"

        self.predictions_text.insert(1.0, text)

    def plot_classification_confidence(self, probabilities, categories):
        """Построение графика уверенности"""
        self.confidence_ax.clear()

        probs = probabilities.cpu().numpy() * 100
        cats = categories.cpu().numpy()

        # Получаем названия классов
        labels = []
        for cat in cats:
            class_name = self.idx_to_class.get(cat, f"Class {cat}")
            # Обрезаем длинные названия
            if len(class_name) > 20:
                class_name = class_name[:17] + "..."
            labels.append(class_name)

        # Цвета для столбцов
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

        # Создаем горизонтальную столбчатую диаграмму
        y_pos = np.arange(len(probs))
        bars = self.confidence_ax.barh(y_pos, probs, color=colors[:len(probs)])

        # Настройки графика
        self.confidence_ax.set_yticks(y_pos)
        self.confidence_ax.set_yticklabels(labels)
        self.confidence_ax.set_xlabel('Уверенность (%)')
        self.confidence_ax.set_title('Топ предсказаний EfficientNet B0')
        self.confidence_ax.set_xlim([0, 100])

        # Добавляем значения на столбцы
        for bar, prob in zip(bars, probs):
            width = bar.get_width()
            self.confidence_ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                                    f'{prob:.1f}%', va='center', fontsize=10)

        # Настраиваем внешний вид
        self.confidence_ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        self.confidence_ax.set_facecolor('#f8f9fa')

        self.confidence_figure.tight_layout()
        self.confidence_canvas.draw()

    def update_status(self, message):
        """Обновление статус-бара"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_bar.config(text=f"[{timestamp}] {message}")
