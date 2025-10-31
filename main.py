import os
import io
import sys
import sqlite3
import zipfile
import tempfile
import traceback

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt
from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QComboBox, QTabWidget, QTextEdit, QSplitter, QTableView,
    QMessageBox
)

# matplotlib / seaborn
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import seaborn as sns


# ======== простой сервис доступа к данным =========
class DataService:
    def __init__(self, db_path=":memory:", log_fn=None):
        self.conn = sqlite3.connect(db_path)
        self.log = log_fn if log_fn is not None else print

    def _normalize_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = []
        for c in df.columns:
            c = str(c).strip().replace(" ", "_").replace("-", "_")
            cols.append(c)
        df = df.copy()
        df.columns = cols
        return df

    def write_df(self, table: str, df: pd.DataFrame):
        df = self._normalize_cols(df)
        self.log(f"Сохраняю в таблицу '{table}', строк: {len(df)}")
        df.to_sql(table, self.conn, if_exists="replace", index=False)

    def load_csv(self, path: str, table_name: str | None = None):
        if table_name is None:
            table_name = os.path.splitext(os.path.basename(path))[0]
        self.log(f"Читаю CSV: {path}")
        df = pd.read_csv(path)
        self.write_df(table_name, df)
        return table_name, df

    def load_excel(self, path: str):
        self.log(f"Читаю Excel: {path}")
        xls = pd.ExcelFile(path)
        res = []
        base = os.path.splitext(os.path.basename(path))[0]
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            t = f"{base}_{sheet}"
            self.write_df(t, df)
            res.append((t, df))
        return res

    def load_zip(self, path: str):
        self.log(f"Читаю ZIP: {path}")
        out = []
        with zipfile.ZipFile(path) as zf, tempfile.TemporaryDirectory() as tmp:
            zf.extractall(tmp)
            for root, _, files in os.walk(tmp):
                for name in files:
                    fp = os.path.join(root, name)
                    low = name.lower()
                    try:
                        if low.endswith(".csv"):
                            out.append(self.load_csv(fp))
                        elif low.endswith(".xlsx") or low.endswith(".xls"):
                            out.extend(self.load_excel(fp))
                    except Exception as e:
                        self.log(f"Ошибка в файле {name}: {e}")
        return out

    def list_tables(self):
        q = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        return [r[0] for r in self.conn.execute(q).fetchall()]

    def get_df(self, table: str) -> pd.DataFrame:
        return pd.read_sql_query(f"SELECT * FROM '{table}';", self.conn)

    def head(self, table: str, n: int = 200) -> pd.DataFrame:
        return pd.read_sql_query(f"SELECT * FROM '{table}' LIMIT {n};", self.conn)

    def describe(self, table: str) -> pd.DataFrame:
        df = self.get_df(table)
        return df.describe(include='all').T

    def numeric_columns(self, table: str):
        df = self.get_df(table)
        cols = []
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
        return cols



class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        super().__init__(parent)
        self.df = df

    def rowCount(self, parent=None):
        return len(self.df.index)

    def columnCount(self, parent=None):
        return len(self.df.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            val = self.df.iat[index.row(), index.column()]
            if isinstance(val, float):
                return f"{val:.6g}"
            return str(val)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self.df.columns):
                return str(self.df.columns[section])
        else:
            if 0 <= section < len(self.df.index):
                return str(self.df.index[section])
        return None

    def set_df(self, df: pd.DataFrame):
        self.beginResetModel()
        self.df = df.copy()
        self.endResetModel()



class MplCanvas(FigureCanvas):
    def __init__(self, w=7, h=5, dpi=100):
        self.fig = Figure(figsize=(w, h), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gotiscale")
        self.resize(1100, 750)

        # лог на отдельной вкладке
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        def log(msg):
            self.log_box.append(str(msg))
            print(msg)

        self.log = log
        self.svc = DataService(log_fn=self.log)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # вкладки
        self._init_tab1()
        self._init_tab2()
        self._init_tab3()
        self._init_tab4()
        self._init_tab5()

        self._refresh_tables()

    # ---- вкладка 1: статистика ----
    def _init_tab1(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        top = QHBoxLayout()
        self.btn_load = QPushButton("Загрузить датасет…")
        self.btn_load.clicked.connect(self.on_load)
        top.addWidget(self.btn_load)
        top.addStretch(1)

        self.cb_tables1 = QComboBox()
        self.cb_tables1.currentTextChanged.connect(self.on_table_changed)
        top.addWidget(QLabel("Таблица:"))
        top.addWidget(self.cb_tables1)

        lay.addLayout(top)

        split = QSplitter(Qt.Orientation.Vertical)

        self.view_head = QTableView()
        self.model_head = PandasModel(pd.DataFrame())
        self.view_head.setModel(self.model_head)
        split.addWidget(self.view_head)

        self.view_desc = QTableView()
        self.model_desc = PandasModel(pd.DataFrame())
        self.view_desc.setModel(self.model_desc)
        split.addWidget(self.view_desc)

        split.setSizes([400, 300])
        lay.addWidget(split)

        self.tabs.addTab(w, "Статистика")

    def on_table_changed(self, table):
        if not table:
            return
        try:
            self.log(f"Выбрали таблицу: {table}")
            head = self.svc.head(table, 200)
            self.model_head.set_df(head)

            desc = self.svc.describe(table)
            desc = desc.reset_index().rename(columns={'index': 'metric'})
            self.model_desc.set_df(desc)

            # синхронно выберем эту таблицу на других вкладках
            for cb in (self.cb_tables2, self.cb_tables3, self.cb_tables4):
                i = cb.findText(table)
                if i >= 0:
                    cb.setCurrentIndex(i)
        except Exception as e:
            self.err("Ошибка обновления статистики", e)

    def on_load(self):
        dlg = QFileDialog(self, "Файл датасета")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        dlg.setNameFilter("Datasets (*.zip *.csv *.xlsx *.xls)")
        if dlg.exec():
            path = dlg.selectedFiles()[0]
            try:
                if path.lower().endswith(".zip"):
                    loaded = self.svc.load_zip(path)
                elif path.lower().endswith(".csv"):
                    loaded = [self.svc.load_csv(path)]
                elif path.lower().endswith(".xlsx") or path.lower().endswith(".xls"):
                    loaded = self.svc.load_excel(path)
                else:
                    raise ValueError("Неизвестный формат")
                self.log(f"Загружено таблиц: {len(loaded)}")
                self._refresh_tables()
                if loaded:
                    first = loaded[0][0]
                    self._select_table_everywhere(first)
            except Exception as e:
                self.err("Ошибка загрузки", e)

    def _refresh_tables(self):
        tabs = self.svc.list_tables()
        for cb in (getattr(self, "cb_tables1", None),
                   getattr(self, "cb_tables2", None),
                   getattr(self, "cb_tables3", None),
                   getattr(self, "cb_tables4", None)):
            if cb is None:
                continue
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(tabs)
            cb.blockSignals(False)

    def _select_table_everywhere(self, table):
        for cb in (self.cb_tables1, self.cb_tables2, self.cb_tables3, self.cb_tables4):
            i = cb.findText(table)
            if i >= 0:
                cb.setCurrentIndex(i)


    def _init_tab2(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        top = QHBoxLayout()
        self.cb_tables2 = QComboBox()
        self.cb_tables2.currentTextChanged.connect(self.draw_pairplot)
        top.addWidget(QLabel("Таблица:"))
        top.addWidget(self.cb_tables2)

        btn = QPushButton("Построить")
        btn.clicked.connect(self.draw_pairplot)
        top.addWidget(btn)
        top.addStretch(1)
        lay.addLayout(top)

        self.canvas2 = MplCanvas(8, 6, 100)
        lay.addWidget(self.canvas2)

        self.tabs.addTab(w, "Корреляции")

    def draw_pairplot(self):
        table = self.cb_tables2.currentText()
        if not table:
            return
        try:
            df = self.svc.get_df(table)
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] < 2:
                self.info("Нужны хотя бы 2 числовых столбца")
                return
            self.log(f"pairplot для '{table}' (числовых столбцов: {num.shape[1]})")

            plt.close('all')
            g = sns.pairplot(num, corner=True, diag_kind="hist")
            buf = io.BytesIO()
            g.fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)

            self.canvas2.fig.clear()
            ax = self.canvas2.fig.add_subplot(111)
            img = plt.imread(buf)
            ax.imshow(img)
            ax.axis("off")
            self.canvas2.draw()
        except Exception as e:
            self.err("Ошибка построения pairplot", e)

    # ---- вкладка 3: тепловая карта ----
    def _init_tab3(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        top = QHBoxLayout()
        self.cb_tables3 = QComboBox()
        self.cb_tables3.currentTextChanged.connect(self.draw_heatmap)
        top.addWidget(QLabel("Таблица:"))
        top.addWidget(self.cb_tables3)

        btn = QPushButton("Построить")
        btn.clicked.connect(self.draw_heatmap)
        top.addWidget(btn)
        top.addStretch(1)
        lay.addLayout(top)

        self.canvas3 = MplCanvas(8, 6, 100)
        lay.addWidget(self.canvas3)

        self.tabs.addTab(w, "Тепловая карта")

    def draw_heatmap(self):
        table = self.cb_tables3.currentText()
        if not table:
            return
        try:
            df = self.svc.get_df(table)
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] < 2:
                self.info("Нужны хотя бы 2 числовых столбца")
                return
            corr = num.corr(numeric_only=True)

            self.log(f"heatmap для '{table}'")
            self.canvas3.fig.clear()
            ax = self.canvas3.fig.add_subplot(111)
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title(f"Корреляции ({table})")
            self.canvas3.fig.tight_layout()
            self.canvas3.draw()
        except Exception as e:
            self.err("Ошибка построения heatmap", e)

    # ---- вкладка 4: линейный график ----
    def _init_tab4(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        top = QHBoxLayout()
        self.cb_tables4 = QComboBox()
        self.cb_tables4.currentTextChanged.connect(self.update_numeric_cols)
        top.addWidget(QLabel("Таблица:"))
        top.addWidget(self.cb_tables4)

        self.cb_num_col = QComboBox()
        top.addWidget(QLabel("Числовой столбец:"))
        top.addWidget(self.cb_num_col)

        btn = QPushButton("Построить")
        btn.clicked.connect(self.draw_line)
        top.addWidget(btn)
        top.addStretch(1)
        lay.addLayout(top)

        self.canvas4 = MplCanvas(8, 6, 100)
        lay.addWidget(self.canvas4)

        self.tabs.addTab(w, "Линейный график")

    def update_numeric_cols(self):
        table = self.cb_tables4.currentText()
        self.cb_num_col.clear()
        if not table:
            return
        try:
            cols = self.svc.numeric_columns(table)
            self.cb_num_col.addItems(cols)
        except Exception as e:
            self.err("Ошибка получения столбцов", e)

    def draw_line(self):
        table = self.cb_tables4.currentText()
        col = self.cb_num_col.currentText()
        if not table or not col:
            return
        try:
            df = self.svc.get_df(table)
            if col not in df.columns:
                self.info("Выберите числовой столбец")
                return
            if not pd.api.types.is_numeric_dtype(df[col]):
                self.info("Столбец не числовой")
                return

            self.log(f"line plot: таблица='{table}', столбец='{col}'")
            self.canvas4.fig.clear()
            ax = self.canvas4.fig.add_subplot(111)
            ax.plot(df.index.values, df[col].values, linewidth=2)
            ax.set_xlabel("Index")
            ax.set_ylabel(col)
            ax.set_title(f"{col} — {table}")
            self.canvas4.fig.tight_layout()
            self.canvas4.draw()
        except Exception as e:
            self.err("Ошибка построения line plot", e)

    # ---- вкладка 5: лог ----
    def _init_tab5(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(self.log_box)
        self.tabs.addTab(w, "Ход работы")

    # ---- сообщения ----
    def err(self, title, ex):
        tb = traceback.format_exc()
        self.log(f"[ОШИБКА] {title}: {ex}\n{tb}")
        QMessageBox.critical(self, title, f"{ex}")

    def info(self, msg):
        self.log(f"[ИНФО] {msg}")
        QMessageBox.information(self, "Информация", msg)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
