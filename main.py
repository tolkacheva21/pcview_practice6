import tkinter as tk
from image_classifier import ImageClassifier
from tkinter import messagebox
import time

root = tk.Tk()
app = ImageClassifier(root)

# Обработка закрытия окна
def on_closing():
    if app.is_training:
        if messagebox.askyesno("Обучение в процессе",
                               "Обучение все еще выполняется. Вы уверены, что хотите выйти?"):
            app.is_training = False
            time.sleep(0.5)  # Даем время для остановки
            root.destroy()
    else:
        if messagebox.askokcancel("Выход", "Вы уверены, что хотите выйти?"):
            root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
