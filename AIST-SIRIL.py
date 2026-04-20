##############################################
# AIST – Siril Plugin (AITS-style shell)
# Port complet al pipeline-ului AIST pe imaginea curentă din Siril
##############################################

import sys
import os
import webbrowser

try:
    import sirilpy as s
    from sirilpy import LogColor
except ImportError:
    print("Error: sirilpy module not found.")
    sys.exit(1)

# instalăm ce ne trebuie
s.ensure_installed("numpy", "PyQt6", "opencv-python-headless")

import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QSlider, QSpacerItem, QSizePolicy,
    QGridLayout, QCheckBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings
from PyQt6.QtGui import QImage, QPixmap

# ---------------------
#  THEME & STYLING
# ---------------------
AITS_STYLE = """
QWidget { background-color: #1b1b1b; color: #cccccc; font-size: 10pt; }
QLabel { color: #cccccc; }

QPushButton {
    background-color: #2a2a2a;
    border: 1px solid #804040;
    border-radius: 6px;
    padding: 6px;
}
QPushButton:hover { background-color: #00994d; }
QPushButton#ProcessButton { background-color: #2a2a2a; border: 1px solid #804040; }
QPushButton#ProcessButton:hover { background-color: #00994d; }
QPushButton#CloseButton { background-color: #2a2a2a; border: 1px solid #804040; }
QPushButton#CloseButton:hover { background-color: #00994d; }

QSlider::groove:horizontal { height: 6px; background: #444; }
QSlider::handle:horizontal {
    background: #2ec4c7;
    width: 12px;
    margin: -5px 0;
}

QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #2ec4c7;
    background-color: #1b1b1b;
}
QCheckBox::indicator:checked {
    background-color: #2ec4c7;
}
"""

VERSION = "1.3"

# =============================================================================
#  CORE: NORMALIZARE INPUT
# =============================================================================

class AITS_LVCore:
    @staticmethod
    def normalize_input(img_data):
        """Ensures float32 0-1 range."""
        input_dtype = img_data.dtype
        img_float = img_data.astype(np.float32)
        if np.issubdtype(input_dtype, np.integer):
            if input_dtype == np.uint8:
                return img_float / 255.0
            elif input_dtype == np.uint16:
                return img_float / 65535.0
            else:
                return img_float / float(np.iinfo(input_dtype).max)
        elif np.issubdtype(input_dtype, np.floating):
            current_max = np.max(img_data)
            if current_max <= 1.0 + 1e-5:
                return img_float
            if current_max <= 65535.0:
                return img_float / 65535.0
            return img_float
        return img_float

# =============================================================================
#  AIST PIPELINE – PORTAT 1:1
# =============================================================================

def aist_auto_stf(img):
    img = img.astype(np.float32)
    p1 = np.percentile(img, 0.2)
    p2 = np.percentile(img, 99.65)
    img = (img - p1) / (p2 - p1 + 1e-6)
    img = np.clip(img, 0, 1)
    img = np.power(img, 0.6)
    return (img * 255).astype(np.uint8)

def aist_auto_white_balance(img):
    # img: HWC, RGB, float32 0–1
    b, g, r = cv2.split(img.astype(np.float32))  # ordinea canalelor nu contează la medii
    avg = (np.mean(b) + np.mean(g) + np.mean(r)) / 3.0
    b *= avg / (np.mean(b) + 1e-6)
    g *= avg / (np.mean(g) + 1e-6)
    r *= avg / (np.mean(r) + 1e-6)
    return cv2.merge([b, g, r])

def aist_stretch(img, black_slider, mid_slider, white_slider,
                 autostretch_slider, auto_stretch_checked, highlight_slider):
    img = img.astype(np.float32)

    if auto_stretch_checked:
        f = autostretch_slider / 100.0
        black = np.percentile(img, 0.25)
        white = np.percentile(img, 99 + f)
    else:
        maxv = np.max(img)
        black = black_slider / 100.0 * maxv
        white = white_slider / 100.0 * maxv

    gamma = max(0.1, mid_slider / 50.0)

    img = (img - black) / (white - black + 1e-6)
    img = np.clip(img, 0, 1)

    # luminance stretch
    luma = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    l = luma / (luma + (1.0 - luma) * gamma)
    img = img * (l[..., None] / (luma[..., None] + 1e-6))
    img = np.clip(img, 0, 1)

    # highlight protect
    k = highlight_slider / 100.0
    if k > 0:
        img = img / (1.0 + k * img)

    return np.clip(img, 0, 1)

def aist_apply_background(img, bg_slider):
    val = bg_slider
    if val == 0:
        return img
    strength = (val / 100.0) ** 1.5
    bg = np.percentile(img, 20, axis=(0, 1))
    img = img - bg * strength
    return np.clip(img, 0, 1)

def aist_apply_enhance(img, enhance_slider):
    val = enhance_slider
    if val == 0:
        return img
    strength = (val / 100.0) ** 1.5
    blur = cv2.GaussianBlur(img, (0, 0), 2)
    img = cv2.addWeighted(img, 1 + 0.4 * strength, blur, -0.4 * strength, 0)

    luma = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    luma = np.clip(luma, 0, None)
    sat = 1 + 0.3 * strength * np.sqrt(luma)
    img = img * sat[..., None]

    return np.clip(img, 0, 1)

def aist_process_pipeline(img_rgb,
                          auto_wb_checked,
                          auto_stretch_checked,
                          stf_checked,
                          black_slider,
                          mid_slider,
                          white_slider,
                          enhance_slider,
                          bg_slider,
                          highlight_slider,
                          autostretch_slider):
    """
    img_rgb: HWC, RGB, float32 0–1
    return: HWC, RGB, float32 0–1
    """
    img = img_rgb.astype(np.float32)

    if auto_wb_checked:
        img = aist_auto_white_balance(img)

    img = aist_stretch(
        img,
        black_slider=black_slider,
        mid_slider=mid_slider,
        white_slider=white_slider,
        autostretch_slider=autostretch_slider,
        auto_stretch_checked=auto_stretch_checked,
        highlight_slider=highlight_slider
    )

    img = aist_apply_background(img, bg_slider)
    img = aist_apply_enhance(img, enhance_slider)

    return np.clip(img, 0, 1)

# =============================================================================
#  GUI HELPERS
# =============================================================================

class ResetSlider(QSlider):
    def __init__(self, orientation, default_val=0, parent=None):
        super().__init__(orientation, parent)
        self.default_val = default_val
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setValue(self.default_val)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

class FitGraphicsView(QGraphicsView):
    def __init__(self, scene, on_double_click=None, parent=None):
        super().__init__(scene, parent)
        self._on_double_click = on_double_click

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and callable(self._on_double_click):
            self._on_double_click()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

# =============================================================================
#  WORKER THREAD
# =============================================================================

class AISTWorker(QThread):
    result_ready = pyqtSignal(object, object)  # (linear_out, preview8)

    def __init__(self, img_proxy, params):
        super().__init__()
        self.img_proxy = img_proxy  # CHW, RGB, 0–1
        self.p = params

    def run(self):
        # convert CHW -> HWC
        img = np.transpose(self.img_proxy, (1, 2, 0))

        proc = aist_process_pipeline(
            img,
            auto_wb_checked=self.p['auto_wb'],
            auto_stretch_checked=self.p['auto_stretch'],
            stf_checked=self.p['stf'],
            black_slider=self.p['black'],
            mid_slider=self.p['mid'],
            white_slider=self.p['white'],
            enhance_slider=self.p['enhance'],
            bg_slider=self.p['bg'],
            highlight_slider=self.p['highlight'],
            autostretch_slider=self.p['autostretch']
        )

        proc = np.nan_to_num(proc, nan=0.0, posinf=1.0, neginf=0.0)
        proc8 = (proc * 255).astype(np.uint8)

        # întoarcem și varianta float (CHW) pentru consistență
        linear_out = np.transpose(proc, (2, 0, 1))
        self.result_ready.emit(linear_out, proc8)

# =============================================================================
#  MAIN WINDOW
# =============================================================================

class AISTSirilGUI(QMainWindow):
    def __init__(self, siril, app):
        super().__init__()
        self.siril = siril
        self.app = app
        self.setWindowTitle(f"AIST – Siril Plugin v. {VERSION}")
        self.setStyleSheet(AITS_STYLE)
        self.resize(1400, 720)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        self.settings = QSettings("AIST", "SirilPlugin")

        self.img_full = None   # CHW, RGB, 0–1
        self.img_proxy = None  # CHW, RGB, 0–1

        self.debounce = QTimer()
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(150)
        self.debounce.timeout.connect(self.run_worker)

        header_msg = (
            "\n##############################################\n"
            "# AIST – AstroImage Stretch Tool (Siril)\n"
            "# Port complet pe imaginea curentă din Siril\n"
            "##############################################"
        )
        try:
            self.siril.log(header_msg)
        except Exception:
            print(header_msg)

        self.init_ui()
        self.cache_input()

    def init_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        main_layout = QHBoxLayout(main)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)


        # --- LEFT PANEL (slidere + butoane) ---
        left_container = QWidget()
        left_container.setFixedWidth(300)   # ← AICI faci partea stângă mai mică
        left = QVBoxLayout(left_container)
        left.setSpacing(6)

        lbl_title = QLabel("AstroImage Stretch Tool (AIST)")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("color: #2ec4c7; font-size: 16px; font-weight: bold;")
        left.addWidget(lbl_title)

        # === BRAND ===
        brand = QLabel("© 2026 by Lucas V.")
        brand.setStyleSheet("color: #2ec4c7; font-size: 10pt; font-weight: italic;")
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)

        left.addWidget(brand)
      
        # Slidere
        sliders = QGridLayout()
        sliders.setHorizontalSpacing(6)
        sliders.setVerticalSpacing(4)

        # Black
        self.black_slider = ResetSlider(Qt.Orientation.Horizontal, 0)
        self.black_slider.setRange(0, 100)
        self.black_slider.setValue(0)
        self.black_value = QLabel("0")

        # Mid
        self.mid_slider = ResetSlider(Qt.Orientation.Horizontal, 32)
        self.mid_slider.setRange(1, 100)
        self.mid_slider.setValue(32)
        self.mid_value = QLabel("1.00")

        # White
        self.white_slider = ResetSlider(Qt.Orientation.Horizontal, 100)
        self.white_slider.setRange(1, 100)
        self.white_slider.setValue(100)
        self.white_value = QLabel("100")

        # Enhance
        self.enhance_slider = ResetSlider(Qt.Orientation.Horizontal, 50)
        self.enhance_slider.setRange(0, 100)
        self.enhance_slider.setValue(50)

        # Background
        self.bg_slider = ResetSlider(Qt.Orientation.Horizontal, 35)
        self.bg_slider.setRange(0, 100)
        self.bg_slider.setValue(35)

        # Highlight Protect
        self.highlight_slider = ResetSlider(Qt.Orientation.Horizontal, 5)
        self.highlight_slider.setRange(0, 200)
        self.highlight_slider.setValue(5)

        # Stretch Factor
        self.autostretch_slider = ResetSlider(Qt.Orientation.Horizontal, 65)
        self.autostretch_slider.setRange(1, 99)
        self.autostretch_slider.setValue(65)
        self.autostretch_value = QLabel("0.65")
              
        sliders.addItem(QSpacerItem(0, 15), 0, 0)
        sliders.addItem(QSpacerItem(0, 15), 1, 0)
        sliders.addItem(QSpacerItem(0, 15), 2, 0)
        sliders.addItem(QSpacerItem(0, 15), 3, 0)

        sliders.addWidget(QLabel("Black"), 4, 0)
        sliders.addWidget(self.black_slider, 4, 1)
        sliders.addWidget(self.black_value, 4, 2)

        sliders.addWidget(QLabel("Mid"), 5, 0)
        sliders.addWidget(self.mid_slider, 5, 1)
        sliders.addWidget(self.mid_value, 5, 2)

        sliders.addWidget(QLabel("White"), 6, 0)
        sliders.addWidget(self.white_slider, 6, 1)
        sliders.addWidget(self.white_value, 6, 2)
        
        sliders.addItem(QSpacerItem(0, 15), 7, 0)

        sliders.addWidget(QLabel("Enhance"), 8, 0)
        sliders.addWidget(self.enhance_slider, 8, 1)

        sliders.addWidget(QLabel("Background"), 9, 0)
        sliders.addWidget(self.bg_slider, 9, 1)

        sliders.addWidget(QLabel("Highlight Protect"), 10, 0)
        sliders.addWidget(self.highlight_slider, 10, 1)
        
        sliders.addItem(QSpacerItem(0, 15), 11, 0)

        sliders.addWidget(QLabel("Stretch Factor"), 12, 0)
        sliders.addWidget(self.autostretch_slider, 12, 1)
        sliders.addWidget(self.autostretch_value, 12, 2)
                
        left.addStretch()
        left.addLayout(sliders)

        # Checkboxes
        cb_layout = QHBoxLayout()
        self.auto_wb = QCheckBox("Auto WB")
        self.auto_wb.setChecked(True)
        self.auto_stretch = QCheckBox("Auto Stretch")
        self.auto_stretch.setChecked(True)
        self.stf_cb = QCheckBox("STF")
        self.stf_cb.setChecked(False)
        cb_layout.addWidget(self.auto_wb)
        cb_layout.addWidget(self.auto_stretch)
        cb_layout.addWidget(self.stf_cb)
        
        left.addStretch()
        left.addLayout(cb_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_help = QPushButton("Help")
        # self.btn_help.setFixedWidth(24)
        self.btn_help.clicked.connect(self.print_help)

        self.btn_defaults = QPushButton("Reset")
        self.btn_defaults.clicked.connect(self.set_defaults)

        self.btn_close = QPushButton("Close")
        self.btn_close.setObjectName("CloseButton")
        self.btn_close.clicked.connect(self.close)

        self.btn_process = QPushButton("Apply")
        self.btn_process.setObjectName("ProcessButton")
        self.btn_process.clicked.connect(self.process_final)

        btn_layout.addWidget(self.btn_help)
        btn_layout.addWidget(self.btn_defaults)
        btn_layout.addWidget(self.btn_close)
        btn_layout.addWidget(self.btn_process)
        
        left.addStretch()
        left.addLayout(btn_layout)

        main_layout.addWidget(left_container, 0)

        
        # --- RIGHT PANEL (Preview) ---
        right = QVBoxLayout()
        
        tb = QHBoxLayout()
        b_out = QPushButton("-"); b_out.setObjectName("ZoomBtn"); b_out.clicked.connect(self.zoom_out)
        b_fit = QPushButton("Fit"); b_fit.setObjectName("ZoomBtn"); b_fit.clicked.connect(self.fit_view)
        b_11 = QPushButton("1:1"); b_11.setObjectName("ZoomBtn"); b_11.clicked.connect(self.zoom_1to1)
        b_in = QPushButton("+"); b_in.setObjectName("ZoomBtn"); b_in.clicked.connect(self.zoom_in)
        lbl_hint = QLabel("Preview: Double-click to fit")
        lbl_hint.setStyleSheet("color: #00ff00; font-size: 10pt; font-style: italic; margin-left: 10px;")
        
        self.btn_coffee = QPushButton("☕")
        self.btn_coffee.setObjectName("CoffeeButton")
        self.btn_coffee.setToolTip("I hope Like it!")
        self.btn_coffee.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_coffee.clicked.connect(lambda: webbrowser.open("https://mdci.ro/aipt.php"))


        self.chk_ontop = QCheckBox("On Top")
        self.chk_ontop.setChecked(True)
        self.chk_ontop.toggled.connect(self.toggle_ontop)
        
        tb.addWidget(b_out); tb.addWidget(b_fit); tb.addWidget(b_11); tb.addWidget(b_in); tb.addWidget(lbl_hint)
        tb.addStretch(); tb.addWidget(self.btn_coffee); tb.addWidget(self.chk_ontop)
        right.addLayout(tb)
        
        self.scene = QGraphicsScene()
        self.view = FitGraphicsView(self.scene, on_double_click=self.fit_view)
        self.view.setStyleSheet("background-color: #151515; border: none;")
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        right.addWidget(self.view)

        self.pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pix_item)
        main_layout.addLayout(right)

        # conexiuni slidere
        for s in [
            self.black_slider,
            self.mid_slider,
            self.white_slider,
            self.enhance_slider,
            self.bg_slider,
            self.highlight_slider,
            self.autostretch_slider
        ]:
            s.valueChanged.connect(self.update_values)
            s.valueChanged.connect(self.trigger_update)

        self.auto_wb.stateChanged.connect(self.trigger_update)
        self.auto_stretch.stateChanged.connect(self.trigger_update)
        self.stf_cb.stateChanged.connect(self.trigger_update)

        self.update_values()

    # --- LOGIC ---
    def cache_input(self):
        try:
            if not self.siril.connected:
                self.siril.connect()
            with self.siril.image_lock():
                img = self.siril.get_image_pixeldata()
            if img is None:
                return

            img = AITS_LVCore.normalize_input(img)
            # Siril: (C,H,W). Asigurăm RGB
            if img.ndim == 2:
                img = np.array([img, img, img])
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)

            self.img_full = img

            h, w = img.shape[1], img.shape[2]
            scale = 2048 / max(h, w)
            if scale < 1.0:
                step = int(1 / scale)
                self.img_proxy = img[:, ::step, ::step].copy()
            else:
                self.img_proxy = img.copy()

            self.trigger_update()

        except Exception as e:
            print(f"Input Error: {e}")

    def update_values(self):
        gamma = max(0.1, self.mid_slider.value() / 50.0)
        self.black_value.setText(str(self.black_slider.value()))
        self.mid_value.setText(f"{gamma:.2f}")
        self.white_value.setText(str(self.white_slider.value()))
        f = self.autostretch_slider.value() / 100.0
        self.autostretch_value.setText(f"{f:.2f}")

    def set_defaults(self):
        self.black_slider.setValue(0)
        self.mid_slider.setValue(32)
        self.white_slider.setValue(100)
        self.enhance_slider.setValue(50)
        self.bg_slider.setValue(35)
        self.highlight_slider.setValue(5)
        self.autostretch_slider.setValue(65)
        self.auto_wb.setChecked(True)
        self.auto_stretch.setChecked(True)
        self.stf_cb.setChecked(False)

    def trigger_update(self):
        if self.img_proxy is None:
            return
        self.debounce.start()

    def run_worker(self):
        if self.img_proxy is None:
            return

        p = {
            'auto_wb': self.auto_wb.isChecked(),
            'auto_stretch': self.auto_stretch.isChecked(),
            'stf': self.stf_cb.isChecked(),
            'black': self.black_slider.value(),
            'mid': self.mid_slider.value(),
            'white': self.white_slider.value(),
            'enhance': self.enhance_slider.value(),
            'bg': self.bg_slider.value(),
            'highlight': self.highlight_slider.value(),
            'autostretch': self.autostretch_slider.value()
        }

        self.worker = AISTWorker(self.img_proxy, p)
        self.worker.result_ready.connect(self.update_display)
        self.worker.start()

    def update_display(self, linear, preview8):
        # preview8: HWC, uint8, RGB
        disp = preview8
        h, w, c = disp.shape
        qimg = QImage(disp.data.tobytes(), w, h, c * w, QImage.Format.Format_RGB888)
        self.pix_item.setPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(0, 0, w, h)

        if self.view.transform().isIdentity():
            self.fit_view()

    def process_final(self):
        if self.img_full is None:
            return
        self.setEnabled(False)

        try:
            # CHW -> HWC
            img = np.transpose(self.img_full, (1, 2, 0))

            proc = aist_process_pipeline(
                img,
                auto_wb_checked=self.auto_wb.isChecked(),
                auto_stretch_checked=self.auto_stretch.isChecked(),
                stf_checked=self.stf_cb.isChecked(),
                black_slider=self.black_slider.value(),
                mid_slider=self.mid_slider.value(),
                white_slider=self.white_slider.value(),
                enhance_slider=self.enhance_slider.value(),
                bg_slider=self.bg_slider.value(),
                highlight_slider=self.highlight_slider.value(),
                autostretch_slider=self.autostretch_slider.value()
            )

            proc = np.nan_to_num(proc, nan=0.0, posinf=1.0, neginf=0.0)
            out = np.transpose(proc, (2, 0, 1)).astype(np.float32)

            filename = "AIST_Siril_Stretch.fit"
            path = os.path.join(os.getcwd(), filename)
            safe_path = path.replace(os.sep, '/')

            with self.siril.image_lock():
                self.siril.set_image_pixeldata(out)

            self.siril.cmd(f'save "{safe_path}"')
            self.siril.cmd(f'load "{safe_path}"')
            self.siril.log(f"AIST: Saved {filename} & Loaded.", LogColor.GREEN)
            self.close()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
        finally:
            self.setEnabled(True)

    def print_help(self):
        msg = [
            "==========================================================================",
            "   AIST – AstroImage Stretch Tool (Siril Plugin)",
            "   Pipeline: Auto WB, Stretch, Background, Enhance",
            "   Sliders identical to the AIST standalone application.",
            "==========================================================================",
            "1) Open an image (ideally linear) in Siril",
            "2) Run the script: run python aist-siril.py",
            "3) Adjust the sliders (Black/Mid/White/Enhance/Background/Highlight/Stretch).",
            "4) Auto WB / Auto Stretch / STF as in AIST.",
            "5) APPLY writes AIST_Siril_Stretch.fit and reloads it in Siril.",
            "=========================================================================="
        ]
        try:
            for l in msg:
                txt = l if l.strip() else " "
                self.siril.log(txt)
        except Exception:
            print("\n".join(msg))

    def toggle_ontop(self, checked):
        if checked:
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        self.show()

    def zoom_in(self):
        self.view.scale(1.2, 1.2)
    def zoom_out(self):
        self.view.scale(1/1.2, 1/1.2)
    def zoom_1to1(self):
        self.view.resetTransform()
    def fit_view(self):
        self.view.fitInView(self.pix_item, Qt.AspectRatioMode.KeepAspectRatio)

def main():
    app = QApplication(sys.argv)
    siril = s.SirilInterface()
    try:
        siril.connect()
    except Exception:
        pass
    gui = AISTSirilGUI(siril, app)
    gui.show()
    app.exec()

if __name__ == "__main__":
    main()