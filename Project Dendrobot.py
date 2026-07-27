"""
DENDROBOT (Dendrogram BOT) - GUI Hierarchical Clustering (Euclidean)
- Import data dari Excel (xlsx/xls) atau input manual
- Pilih kolom variabel (untuk Excel)
- Pilih linkage: single / average / complete
- Tampilkan dendrogram (SciPy) + Elbow Plot
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from typing import List, Optional
from PIL import Image, ImageTk
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


# ----------------------------- Utilities -----------------------------

def safe_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dataframe columns to numeric; raise ValueError if any invalid values appear."""
    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.isnull().any().any():
        raise ValueError("Kolom yang dipilih mengandung data non-numerik / tidak valid.")
    return numeric


def plot_dendrogram(data: pd.DataFrame, labels: List[str], method: str, title: str) -> None:
    """Compute linkage and show dendrogram."""
    if method not in {"single", "average", "complete"}:
        raise ValueError("Metode linkage tidak valid. Pilih: single / average / complete")

    linked = linkage(data.values, method=method)

    plt.figure(figsize=(12, 8))
    dendrogram(
        linked,
        orientation="top",
        labels=labels,
        distance_sort="descending",
        show_leaf_counts=True,
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Samples", fontsize=12)
    plt.ylabel("Distance (Euclidean)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def _wcss_for_labels(data: np.ndarray, labels: np.ndarray) -> float:
    """
    Hitung WCSS (within-cluster sum of squares) untuk partisi cluster tertentu.
    data  : array (n_samples, n_features)
    labels: array (n_samples,) berisi nomor cluster (1,2,...)
    """
    wcss = 0.0
    for lab in np.unique(labels):
        cluster_points = data[labels == lab]
        if len(cluster_points) == 0:
            continue
        center = cluster_points.mean(axis=0)
        wcss += ((cluster_points - center) ** 2).sum()
    return float(wcss)


def plot_elbow_hierarchical(
    data: pd.DataFrame,
    method: str = "average",
    k_min: int = 1,
    k_max: int = 10,
    title: str = "Elbow Plot (Hierarchical Clustering)",
):
    """
    Membuat grafik Elbow untuk hierarchical clustering.
    data  : DataFrame numerik yang dipakai untuk clustering
    method: metode linkage ("single", "average", "complete", ...)
    """
    if method not in {"single", "average", "complete"}:
        raise ValueError("Metode linkage tidak valid. Pilih: single / average / complete")

    arr = data.values
    linked = linkage(arr, method=method)

    # jangan paksa k lebih besar dari jumlah sampel
    n_samples = arr.shape[0]
    k_max_eff = max(k_min, min(k_max, n_samples))

    ks = list(range(k_min, k_max_eff + 1))
    wcss_vals = []

    for k in ks:
        labels = fcluster(linked, t=k, criterion="maxclust")
        wcss = _wcss_for_labels(arr, labels)
        wcss_vals.append(wcss)

    plt.figure(figsize=(6, 4))
    plt.plot(ks, wcss_vals, marker="o")
    plt.xticks(ks)
    plt.xlabel("Jumlah cluster (k)")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return ks, wcss_vals


# ----------------------------- App State -----------------------------

@dataclass
class AppState:
    df: Optional[pd.DataFrame] = None
    file_path: Optional[str] = None
    label_column: Optional[str] = None


# ----------------------------- GUI Pages -----------------------------
class BackgroundPage(ttk.Frame):
    """
    Base class: halaman dengan background gambar (fit to window).
    Anak class cukup panggil: super().__init__(..., bg_filename="xxx.png")
    lalu tambah tombol via self.canvas.create_window(...)
    """
    def __init__(self, parent, app, bg_filename: str):
        super().__init__(parent)
        self.app = app

        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self._bg_path = os.path.join(base_dir, bg_filename)

        self._bg_raw = Image.open(self._bg_path)
        self._bg_tk = ImageTk.PhotoImage(self._bg_raw)
        self._bg_id = self.canvas.create_image(0, 0, image=self._bg_tk, anchor="nw")

        self.canvas.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        w, h = event.width, event.height
        resized = self._bg_raw.resize((w, h))
        self._bg_tk = ImageTk.PhotoImage(resized)
        self.canvas.itemconfig(self._bg_id, image=self._bg_tk)

        # hook untuk anak class (biar tombol/panel ikut pindah)
        if hasattr(self, "_reposition"):
            self._reposition(w, h)


class StartPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(base_dir, "Page Start.png")

        self.bg_raw = Image.open(img_path)
        self.bg_tk = ImageTk.PhotoImage(self.bg_raw)

        self.bg_id = self.canvas.create_image(0, 0, image=self.bg_tk, anchor="nw")

        self.start_btn = tk.Button(
            self.canvas,
            text="START",
            command=lambda: app.show_page("MenuPage"),
            font=("Segoe UI", 11, "bold"),
            fg="white",
            bg="#0f172a",
            activebackground="#111c36",
            activeforeground="white",
            bd=0,
            padx=18,
            pady=8,
            cursor="hand2"
        )

        self.btn_id = self.canvas.create_window(520, 220, window=self.start_btn)
        self.canvas.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        w, h = event.width, event.height

        resized = self.bg_raw.resize((w, h))
        self.bg_tk = ImageTk.PhotoImage(resized)
        self.canvas.itemconfig(self.bg_id, image=self.bg_tk)

        x = int(w * 0.70)
        y = int(h * 0.62)
        self.canvas.coords(self.btn_id, x, y)


class MenuPage(BackgroundPage):
    def __init__(self, parent, app):
        super().__init__(parent, app, bg_filename="Page Menu.png")

        self.btn_excel = tk.Button(
            self.canvas,
            text="IMPORT\nEXCEL",
            command=lambda: app.show_page("ExcelPage"),
            font=("Segoe UI", 10, "bold"),
            fg="#ffd166",
            bg="#0b1022",
            activebackground="#0f1733",
            activeforeground="#ffd166",
            bd=2,
            relief="solid",
            padx=8,
            pady=16,
            cursor="hand2"
        )

        self.btn_manual = tk.Button(
            self.canvas,
            text="Manual\nINPUT",
            command=lambda: app.show_page("ManualSetupPage"),
            font=("Segoe UI", 10, "bold"),
            fg="#ffd166",
            bg="#0b1022",
            activebackground="#0f1733",
            activeforeground="#ffd166",
            bd=2,
            relief="solid",
            padx=8,
            pady=16,
            cursor="hand2"
        )

        self.btn_back = tk.Button(
            self.canvas,
            text="Back",
            command=lambda: app.show_page("StartPage"),
            font=("Segoe UI", 9, "bold"),
            fg="white",
            bg="#0f172a",
            activebackground="#111c36",
            activeforeground="white",
            bd=0,
            padx=10,
            pady=6,
            cursor="hand2"
        )

        self.excel_id = self.canvas.create_window(520, 200, window=self.btn_excel)
        self.manual_id = self.canvas.create_window(520, 290, window=self.btn_manual)
        self.back_id = self.canvas.create_window(60, 30, window=self.btn_back)

    def _reposition(self, w, h):
        x = int(w * 0.5)
        y1 = int(h * 0.48)
        y2 = int(h * 0.68)
        self.canvas.coords(self.excel_id, x, y1)
        self.canvas.coords(self.manual_id, x, y2)
        self.canvas.coords(self.back_id, 60, 30)


class ExcelPage(BackgroundPage):
    def __init__(self, parent, app):
        super().__init__(parent, app, bg_filename="Page Input.png")
        self.app = app

        style = ttk.Style(self)
        dark_blue = "#131b59"

        style.configure("Excel.TFrame", background=dark_blue)
        style.configure("Excel.TLabel", background=dark_blue, foreground="white")
        style.configure("ExcelHeader.TLabel", background=dark_blue, foreground="white")

        self.content = ttk.Frame(self.canvas, style="Excel.TFrame")
        self.content.columnconfigure(0, weight=1)
        self.content.columnconfigure(1, weight=1)

        self.content_id = self.canvas.create_window(
            80, 60, anchor="nw", window=self.content
        )

        header = ttk.Label(
            self.content,
            text="Import Excel",
            font=("Segoe UI", 16, "bold"),
            style="ExcelHeader.TLabel",
        )
        header.grid(row=0, column=0, columnspan=2, pady=(18, 8), padx=16, sticky="w")

        self.path_var = tk.StringVar(value="Belum ada file dipilih.")
        path_lbl = ttk.Label(
            self.content, textvariable=self.path_var, wraplength=520, style="Excel.TLabel"
        )
        pick_btn = ttk.Button(self.content, text="Pilih File Excel...", command=self.pick_file)

        path_lbl.grid(row=1, column=0, columnspan=2, pady=(0, 10), padx=16, sticky="w")
        pick_btn.grid(row=2, column=0, pady=6, padx=16, sticky="w")

        ttk.Label(
            self.content,
            text="Kolom label (nama sampel/wilayah):",
            style="Excel.TLabel"
        ).grid(row=3, column=0, padx=16, pady=(12, 4), sticky="w")

        self.label_combo = ttk.Combobox(self.content, state="readonly", values=[])
        self.label_combo.grid(row=4, column=0, padx=16, pady=4, sticky="ew")

        ttk.Label(
            self.content,
            text="Pilih kolom variabel (numerik) untuk clustering:",
            style="Excel.TLabel"
        ).grid(row=5, column=0, padx=16, pady=(12, 4), sticky="w")

        self.column_listbox = tk.Listbox(self.content, selectmode=tk.MULTIPLE, height=10)
        self.column_listbox.grid(row=6, column=0, padx=16, pady=4, sticky="nsew")

        scrollbar = ttk.Scrollbar(self.content, orient="vertical", command=self.column_listbox.yview)
        scrollbar.grid(row=6, column=1, padx=(0, 16), pady=4, sticky="ns")
        self.column_listbox.config(yscrollcommand=scrollbar.set)

        ttk.Label(
            self.content,
            text="Metode linkage:",
            style="Excel.TLabel"
        ).grid(row=7, column=0, padx=16, pady=(12, 4), sticky="w")

        self.linkage_combo = ttk.Combobox(
            self.content, state="readonly",
            values=["single", "average", "complete"]
        )
        self.linkage_combo.set("average")
        self.linkage_combo.grid(row=8, column=0, padx=16, pady=4, sticky="w")

        show_btn = ttk.Button(self.content, text="Show Dendrogram",
                              command=self.generate_dendrogram_excel)
        back_btn = ttk.Button(self.content, text="Back",
                              command=lambda: app.show_page("MenuPage"))

        show_btn.grid(row=9, column=0, padx=16, pady=(14, 18),
                      sticky="w", ipadx=10, ipady=4)
        back_btn.grid(row=9, column=0, padx=200, pady=(14, 18),
                      sticky="w", ipadx=10, ipady=4)

        self.content.rowconfigure(6, weight=1)

    def _reposition(self, w, h):
        x = int(w * 0.17)
        y = int(h * 0.12)
        self.canvas.coords(self.content_id, x, y)

    def pick_file(self):
        file_path = filedialog.askopenfilename(
            title="Pilih file Excel",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            df = pd.read_excel(file_path)
            if df.empty:
                raise ValueError("File Excel kosong.")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membaca Excel:\n{e}")
            return

        self.app.state.df = df
        self.app.state.file_path = file_path
        self.path_var.set(file_path)

        cols = list(df.columns)
        self.label_combo["values"] = cols
        self.label_combo.set(cols[0] if cols else "")

        self.column_listbox.delete(0, tk.END)
        for c in cols:
            self.column_listbox.insert(tk.END, c)

    def generate_dendrogram_excel(self):
        try:
            df = self.app.state.df
            if df is None:
                raise ValueError("Silakan pilih file Excel terlebih dahulu.")

            selected_idx = list(self.column_listbox.curselection())
            selected_columns = [self.column_listbox.get(i) for i in selected_idx]
            if not selected_columns:
                raise ValueError("Tidak ada kolom variabel yang dipilih untuk clustering.")

            label_col = self.label_combo.get().strip()
            if not label_col:
                raise ValueError("Pilih kolom label terlebih dahulu.")
            if label_col not in df.columns:
                raise ValueError("Kolom label tidak ditemukan di file.")

            selected_data = safe_to_numeric(df[selected_columns])
            labels = df[label_col].astype(str).tolist()
            linkage_method = self.linkage_combo.get()

            # 1) Dendrogram
            plot_dendrogram(
                data=selected_data,
                labels=labels,
                method=linkage_method,
                title=f"Hierarchical Clustering Dendrogram ({linkage_method.title()} Linkage)"
            )

            # 2) Elbow plot
            plot_elbow_hierarchical(
                data=selected_data,
                method=linkage_method,
                k_min=1,
                k_max=10,
                title=f"Elbow Plot - {linkage_method.title()} Linkage (Excel)"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan:\n{e}")


# -------------------------------------------------------------------
# MANUAL SETUP PAGE  (jumlah variabel & sampel)
# -------------------------------------------------------------------
class ManualSetupPage(BackgroundPage):
    """Step 1: user chooses number of variables and number of samples."""
    def __init__(self, parent, app):
        super().__init__(parent, app, bg_filename="Page Input.png")
        self.app = app

        style = ttk.Style(self)
        dark_blue = "#131b59"

        style.configure("ManualSetup.TFrame", background=dark_blue)
        style.configure("ManualSetup.TLabel", background=dark_blue, foreground="white")
        style.configure("ManualSetupHeader.TLabel",
                        background=dark_blue, foreground="white")

        self.content = ttk.Frame(self.canvas, style="ManualSetup.TFrame")
        self.content_id = self.canvas.create_window(
            100, 100, anchor="nw", window=self.content
        )

        header = ttk.Label(
            self.content,
            text="Manual Data Input",
            font=("Segoe UI", 16, "bold"),
            style="ManualSetupHeader.TLabel"
        )
        header.pack(pady=(20, 10), anchor="w", padx=20)

        form = ttk.Frame(self.content, style="ManualSetup.TFrame")
        form.pack(padx=20, anchor="w")

        ttk.Label(form, text="Jumlah variabel:",
                  style="ManualSetup.TLabel").grid(row=0, column=0, pady=6, sticky="w")
        ttk.Label(form, text="Jumlah data/sampel:",
                  style="ManualSetup.TLabel").grid(row=1, column=0, pady=6, sticky="w")

        self.var_count = tk.StringVar(value="4")
        self.sample_count = tk.StringVar(value="10")

        ttk.Entry(form, textvariable=self.var_count, width=10).grid(row=0, column=1, padx=10)
        ttk.Entry(form, textvariable=self.sample_count, width=10).grid(row=1, column=1, padx=10)

        btn_frame = ttk.Frame(self.content, style="ManualSetup.TFrame")
        btn_frame.pack(pady=20, padx=20, anchor="w")

        ttk.Button(btn_frame, text="Next", command=self.go_next).grid(row=0, column=0, padx=(0, 20))
        ttk.Button(btn_frame, text="Back",
                   command=lambda: app.show_page("MenuPage")).grid(row=0, column=1)

    def _reposition(self, w, h):
        x = int(w * 0.35)
        y = int(h * 0.35)
        self.canvas.coords(self.content_id, x, y)

    def go_next(self):
        try:
            v = int(self.var_count.get())
            n = int(self.sample_count.get())
            if v <= 0 or n <= 0:
                raise ValueError("Jumlah variabel dan jumlah data harus > 0.")
            if v > 20:
                raise ValueError("Jumlah variabel terlalu besar (maks 20).")
            if n > 200:
                raise ValueError("Jumlah data terlalu besar (maks 200).")

            self.app.manual_var_count = v
            self.app.manual_sample_count = n
            self.app.show_page("ManualVarNamesPage")
        except Exception as e:
            messagebox.showerror("Error", f"Input tidak valid:\n{e}")


# -------------------------------------------------------------------
# MANUAL VAR NAMES PAGE  (nama variabel)
# -------------------------------------------------------------------
class ManualVarNamesPage(BackgroundPage):
    """Step 2: user inputs variable names."""
    def __init__(self, parent, app):
        super().__init__(parent, app, bg_filename="Page Input.png")
        self.app = app
        self.entries: List[ttk.Entry] = []

        style = ttk.Style(self)
        dark_blue = "#131b59"
        style.configure("ManualNames.TFrame", background=dark_blue)
        style.configure("ManualNames.TLabel", background=dark_blue, foreground="white")
        style.configure("ManualNamesHeader.TLabel",
                        background=dark_blue, foreground="white")

        self.content = ttk.Frame(self.canvas, style="ManualNames.TFrame")
        self.content.columnconfigure(0, weight=1)
        self.content_id = self.canvas.create_window(
            80, 80, anchor="nw", window=self.content
        )

        self.header = ttk.Label(
            self.content,
            text="Nama Variabel",
            font=("Segoe UI", 16, "bold"),
            style="ManualNamesHeader.TLabel",
        )
        self.header.grid(row=0, column=0, pady=(18, 10), padx=16, sticky="w")

        self.container = ttk.Frame(self.content, style="ManualNames.TFrame")
        self.container.grid(row=1, column=0, padx=16, pady=10, sticky="nsew")
        self.container.columnconfigure(0, weight=1)

        self.btn_next = ttk.Button(self.content, text="Next", command=self.go_next)
        self.btn_back = ttk.Button(self.content, text="Back",
                                   command=lambda: app.show_page("ManualSetupPage"))
        self.btn_next.grid(row=2, column=0, padx=16, pady=(10, 18),
                           sticky="w", ipadx=10, ipady=4)
        self.btn_back.grid(row=2, column=0, padx=120, pady=(10, 18),
                           sticky="w", ipadx=10, ipady=4)

        self.content.rowconfigure(1, weight=1)

    def _reposition(self, w, h):
        x = int(w * 0.30)
        y = int(h * 0.30)
        self.canvas.coords(self.content_id, x, y)

    def on_show(self):
        for w in self.container.winfo_children():
            w.destroy()
        self.entries.clear()

        v = getattr(self.app, "manual_var_count", 4)
        ttk.Label(self.container,
                  text=f"Masukkan {v} nama variabel:",
                  style="ManualNames.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )

        for i in range(v):
            row = ttk.Frame(self.container, style="ManualNames.TFrame")
            row.grid(row=i + 1, column=0, sticky="ew", pady=4)
            row.columnconfigure(1, weight=1)
            ttk.Label(row, text=f"Variabel {i+1}:",
                      style="ManualNames.TLabel").grid(
                row=0, column=0, sticky="w", padx=(0, 10)
            )
            e = ttk.Entry(row)
            e.grid(row=0, column=1, sticky="ew")
            e.insert(0, f"Var{i+1}")
            self.entries.append(e)

    def go_next(self):
        try:
            names = [e.get().strip() for e in self.entries]
            if any(not n for n in names):
                raise ValueError("Semua nama variabel harus diisi.")
            if len(set(names)) != len(names):
                raise ValueError("Nama variabel tidak boleh duplikat.")
            self.app.manual_var_names = names
            self.app.show_page("ManualDataEntryPage")
        except Exception as e:
            messagebox.showerror("Error", f"Input tidak valid:\n{e}")


# -------------------------------------------------------------------
# MANUAL DATA ENTRY PAGE  (tabel entry manual)
# -------------------------------------------------------------------
class ManualDataEntryPage(BackgroundPage):
    """Step 3: user enters labels + numeric values; then show dendrogram."""
    def __init__(self, parent, app):
        super().__init__(parent, app, bg_filename="Page Input.png")
        self.app = app

        style = ttk.Style(self)
        dark_blue = "#131b59"
        style.configure("ManualData.TFrame", background=dark_blue)
        style.configure("ManualData.TLabel", background=dark_blue, foreground="white")
        style.configure("ManualDataHeader.TLabel",
                        background=dark_blue, foreground="white")

        self.content = ttk.Frame(self.canvas, style="ManualData.TFrame")
        self.content.columnconfigure(0, weight=1)
        self.content.columnconfigure(1, weight=0)
        self.content.rowconfigure(1, weight=1)
        self.content_id = self.canvas.create_window(
            60, 60, anchor="nw", window=self.content
        )

        header = ttk.Label(
            self.content,
            text="Entry Data Manual",
            font=("Segoe UI", 16, "bold"),
            style="ManualDataHeader.TLabel",
        )
        header.grid(row=0, column=0, pady=(18, 8), padx=16, sticky="w")

        self.data_canvas = tk.Canvas(self.content, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(
            self.content, orient="vertical", command=self.data_canvas.yview
        )
        self.inner = ttk.Frame(self.data_canvas)

        self.inner.bind(
            "<Configure>",
            lambda e: self.data_canvas.configure(scrollregion=self.data_canvas.bbox("all"))
        )
        self.data_canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.data_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.data_canvas.grid(row=1, column=0, padx=16, pady=8, sticky="nsew")
        self.scrollbar.grid(row=1, column=1, pady=8, sticky="ns")

        controls = ttk.Frame(self.content, style="ManualData.TFrame")
        controls.grid(row=2, column=0, padx=16, pady=(6, 18), sticky="w")

        ttk.Label(controls, text="Metode linkage:",
                  style="ManualData.TLabel").grid(
            row=0, column=0, padx=(0, 10), pady=6, sticky="w"
        )
        self.linkage_combo = ttk.Combobox(
            controls, state="readonly",
            values=["single", "average", "complete"], width=12
        )
        self.linkage_combo.set("average")
        self.linkage_combo.grid(row=0, column=1, pady=6, sticky="w")

        show_btn = ttk.Button(controls, text="Show Dendrogram",
                              command=self.show_dendrogram_manual)
        back_btn = ttk.Button(controls, text="Back",
                              command=lambda: app.show_page("ManualVarNamesPage"))
        show_btn.grid(row=0, column=2, padx=(20, 10), pady=6)
        back_btn.grid(row=0, column=3, padx=(0, 10), pady=6)

        self.manual_entries: List[List[tk.Entry]] = []

    def _reposition(self, w, h):
        x = int(w * 0.18)
        y = int(h * 0.18)
        self.canvas.coords(self.content_id, x, y)

    def on_show(self):
        for w in self.inner.winfo_children():
            w.destroy()
        self.manual_entries.clear()

        var_names = getattr(self.app, "manual_var_names", ["Var1", "Var2"])
        v = len(var_names)
        n = getattr(self.app, "manual_sample_count", 10)

        header_row = ttk.Frame(self.inner)
        header_row.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(header_row, text="Label", width=20).grid(row=0, column=0, padx=2)
        for j, name in enumerate(var_names, start=1):
            ttk.Label(header_row, text=name, width=14).grid(row=0, column=j, padx=2)

        for i in range(n):
            row_frame = ttk.Frame(self.inner)
            row_frame.grid(row=i + 1, column=0, sticky="ew", pady=2)

            row_entries: List[tk.Entry] = []

            label_e = ttk.Entry(row_frame, width=22)
            label_e.grid(row=0, column=0, padx=2)
            label_e.insert(0, f"Data{i+1}")
            row_entries.append(label_e)

            for j in range(v):
                e = ttk.Entry(row_frame, width=16)
                e.grid(row=0, column=j + 1, padx=2)
                e.insert(0, "0")
                row_entries.append(e)

            self.manual_entries.append(row_entries)

    def show_dendrogram_manual(self):
        try:
            var_names = getattr(self.app, "manual_var_names", [])
            if not var_names:
                raise ValueError("Nama variabel belum diisi.")

            data: List[List[float]] = []
            labels: List[str] = []

            for row in self.manual_entries:
                label = row[0].get().strip()
                if not label:
                    raise ValueError("Ada label yang kosong pada input manual.")
                labels.append(label)

                row_data = []
                for entry in row[1:]:
                    val_str = entry.get().strip()
                    try:
                        row_data.append(float(val_str))
                    except ValueError:
                        raise ValueError(f"Nilai '{val_str}' bukan angka valid.")
                data.append(row_data)

            df = pd.DataFrame(data, columns=var_names)
            method = self.linkage_combo.get()

            # 1) Dendrogram
            plot_dendrogram(
                data=df,
                labels=labels,
                method=method,
                title=f"Hierarchical Clustering Dendrogram ({method.title()} Linkage)"
            )

            # 2) Elbow Plot
            plot_elbow_hierarchical(
                data=df,
                method=method,
                k_min=1,
                k_max=10,
                title=f"Elbow Plot - {method.title()} Linkage (Manual Input)"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat clustering:\n{e}")


# ----------------------------- Main App -----------------------------

class DendrogenApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DENDROBOT - Dendrogram BOT")
        self.geometry("760x620")
        self.minsize(720, 560)

        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.state = AppState()

        self.manual_var_count = 4
        self.manual_sample_count = 10
        self.manual_var_names: List[str] = []

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self.pages = {}
        for PageCls in (
            StartPage,
            MenuPage,
            ExcelPage,
            ManualSetupPage,
            ManualVarNamesPage,
            ManualDataEntryPage,
        ):
            page = PageCls(container, self)
            name = PageCls.__name__
            self.pages[name] = page
            page.grid(row=0, column=0, sticky="nsew")

        self.show_page("StartPage")

    def show_page(self, name: str):
        page = self.pages[name]
        page.tkraise()
        if hasattr(page, "on_show"):
            page.on_show()


def main():
    app = DendrogenApp()
    app.mainloop()


if __name__ == "__main__":
    main()
