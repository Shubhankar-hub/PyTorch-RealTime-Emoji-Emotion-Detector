import tkinter as tk
from tkinter import messagebox
import subprocess
import sys

root = tk.Tk()
root.title("Emojify App")
root.geometry("300x200")

def run_train():
    subprocess.Popen([sys.executable, "train.py"])
    messagebox.showinfo("Training", "Training started...")

def run_realtime():
    subprocess.Popen([sys.executable, "realtime.py"])

def run_emojify():
    subprocess.Popen([sys.executable, "emojify.py"])

tk.Button(root, text="Train Model", command=run_train, width=20).pack(pady=10)
tk.Button(root, text="Realtime Detection", command=run_realtime, width=20).pack(pady=10)
tk.Button(root, text="Emojify", command=run_emojify, width=20).pack(pady=10)

root.mainloop()
