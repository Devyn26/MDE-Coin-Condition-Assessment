# Updated for F25-06 coin assessment team
# Updated by: Luke Graham
# Date: 2025-10-02

import sys  # system-level utilities (argv, exit)
from pathlib import Path  # filesystem path handling

from PyQt5.QtCore import (  # core Qt classes/signals/threads/settings
    pyqtSlot, Qt, QObject, QThread, pyqtSignal, QPoint, QSize, QSettings  # selected QtCore symbols
)
from PyQt5.QtGui import QPixmap  # image representation for labels
from PyQt5.QtWidgets import (  # Qt widgets and layouts
    QApplication,  # Qt application wrapper
    QWidget,  # base class for windows
    QPushButton,  # clickable button
    QLineEdit,  # single-line text field
    QLabel,  # label for text/images
    QFileDialog,  # file picker dialogs
    QTextEdit,  # multi-line, rich text box
    QGridLayout,  # grid layout manager
    QHBoxLayout,  # horizontal layout
    QVBoxLayout,  # vertical layout
    QSizePolicy,  # size behavior hints
    QSpacerItem,  # empty space item
    QProgressBar,  # progress indicator (spinner when indeterminate)
    QCheckBox,  # checkbox for dark mode toggle
)  # end widgets import

# ---------------- Local modules (keep relative imports for your package) ----------------
try:  # try importing project modules
    from . import patternMatching  # coin-type detection and grading helpers
    #from . import WheatStalkGrader  # Lincoln cent FFT grading
    from . import MorganGrader  # Morgan dollar FFT grading
except Exception:  # if imports fail, degrade gracefully
    patternMatching = None  # sentinel for missing module
    #WheatStalkGrader = None  # sentinel for missing module
    MorganGrader = None  # sentinel for missing module


# ---------------- Simple image label that preserves aspect ratio ----------------
class ImageLabel(QLabel):  # QLabel subclass that scales images while keeping aspect
    def __init__(self, placeholder_path: str = "", parent=None):  # ctor with optional default image
        super().__init__(parent)  # init QLabel
        self._orig_pixmap = QPixmap(placeholder_path) if placeholder_path else QPixmap()  # load placeholder if provided
        self.setAlignment(Qt.AlignCenter)  # center content inside label
        self.setScaledContents(False)  # disable auto-stretch (we scale manually)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # grow/shrink with layout
        if not self._orig_pixmap.isNull():  # if we have a valid pixmap
            self._apply_scaled_pixmap()  # render scaled version

    def set_image_path(self, path: str):  # set image by filepath
        pm = QPixmap(path)  # load pixmap from disk
        if pm.isNull():  # invalid image path
            self.clear()  # clear label content
            self._orig_pixmap = QPixmap()  # reset stored pixmap
        else:  # valid image
            self._orig_pixmap = pm  # store original for scaling
            self._apply_scaled_pixmap()  # apply scaled version

    def resizeEvent(self, event):  # called when the label is resized
        self._apply_scaled_pixmap()  # rescale to new size
        super().resizeEvent(event)  # propagate event

    def _apply_scaled_pixmap(self):  # scale original pixmap to current label size
        if self._orig_pixmap.isNull():  # nothing to draw
            return  # exit early
        target = self.size()  # current label size
        if not target.isValid():  # size not yet valid
            return  # exit early
        scaled = self._orig_pixmap.scaled(  # compute scaled pixmap
            target.width(), target.height(),  # target dimensions
            Qt.KeepAspectRatio, Qt.SmoothTransformation  # preserve aspect and smooth
        )  # end scaled
        super().setPixmap(scaled)  # set the scaled image on QLabel


# ---------------- Worker to run grading off the UI thread ----------------
class GradeWorker(QObject):  # QObject subclass for threaded grading
    finished = pyqtSignal(dict)       # emits {"fft": str, "fm": str}  # signal when done
    failed = pyqtSignal(str)          # emits error message            # signal on failure

    def __init__(self, obverse_path: str, reverse_path: str):  # ctor with image paths
        super().__init__()  # init QObject
        self.obverse_path = obverse_path  # store obverse path
        self.reverse_path = reverse_path  # store reverse path

    def run(self):  # main worker function executed in thread
        try:  # guard all grading logic
            # Decide coin type from reverse
            is_msd = False  # default to non-Morgan
            try:  # attempt coin type detection
                if self.reverse_path and patternMatching:  # need reverse image and module
                    is_msd = patternMatching.imgIsMSD(self.reverse_path)  # detect Morgan vs Lincoln
            except Exception:  # detection failure
                is_msd = False  # fall back to Lincoln path

            # FFT-like grading
            if is_msd and self.reverse_path and MorganGrader:  # Morgan flow if available
                blw, brw, ulw, urw, oss = MorganGrader.gradeMorganSilverDollar(self.reverse_path)  # grade MSD wings
                avg = (blw + brw + ulw + urw) / 4.0  # average feature score
                if avg < 26.194:  # below supported range
                    oss_string = "<3 | Possible Image Error."  # low-range message
                elif avg > 87.487:  # above supported range
                    oss_string = "70 | Above Standard Image Range."  # cap at 70 message
                else:  # within range
                    oss_string = str(round(oss, 3))  # format OSS value
                fft_text = (  # assemble FFT results text
                    f"Lower Left Wing: {round(blw,3)}\n"
                    f"Lower Right Wing: {round(brw,3)}\n"
                    f"Upper Left Wing: {round(ulw,3)}\n"
                    f"Upper Right Wing: {round(urw,3)}\n"
                    f"Estimated Sheldon Scale Grade: {oss_string}"
                )  # end fft_text
            elif self.reverse_path and WheatStalkGrader:  # Lincoln flow if available
                lsg, rsg, oss = WheatStalkGrader.gradeWheatStalkPenny(self.reverse_path)  # grade wheat stalks
                avg = (lsg + rsg) / 2.0  # average stalk score
                if avg < 46.211:  # below supported range
                    oss_string = "<8 | Possible Image Error."  # warn low range
                elif avg > 118.711:  # above supported range
                    oss_string = ">68 | Above Standard Image Range."  # warn high range
                else:  # within range
                    oss_string = str(round(oss, 3))  # format OSS value
                fft_text = (  # assemble FFT results text
                    f"Left Stalk Grade: {round(lsg,3)}\n"
                    f"Right Stalk Grade: {round(rsg,3)}\n"
                    f"Estimated Sheldon Scale Grade: {oss_string}"
                )  # end fft_text
            else:  # neither grader available
                fft_text = "FFT grading unavailable: required module not found."  # module missing message

            # Feature Match grading
            if is_msd and self.reverse_path and patternMatching:  # MSD feature match path
                fm = round(patternMatching.gradeCoin(self.reverse_path, True, False))  # grade from reverse for MSD
                fm_text = str(fm)  # stringify result
            elif self.obverse_path and patternMatching:  # Lincoln feature match path
                fm = round(patternMatching.gradeCoin(self.obverse_path, False, False))  # grade from obverse for Lincoln
                fm_text = str(fm)  # stringify result
            else:  # module not available
                fm_text = "Feature Match unavailable: required module not found."  # module missing message

            self.finished.emit({"fft": fft_text, "fm": fm_text})  # send results back to UI
        except Exception as e:  # any unexpected error
            self.failed.emit(str(e))  # report failure to UI


# ---------------- Main App ----------------
class App(QWidget):  # main window class
    ORG = "F25-06-Team"  # QSettings organization name
    APP = "CoinGraderGUI"  # QSettings application name

    def __init__(self):  # window constructor
        super().__init__()  # init QWidget
        self.title = "Automated Coin Grader"  # window title text

        # state
        self.obverse_path = ""  # selected obverse image path
        self.reverse_path = ""  # selected reverse image path
        self.fft_results = "----"  # last FFT results text
        self.fm_results = "----"  # last feature-match results text
        self.dark_mode = False  # theme toggle flag

        # settings
        self.settings = QSettings(self.ORG, self.APP)  # persistent settings store

        self._build_ui()  # construct all widgets/layouts
        self._restore_settings()  # load saved settings

    # ---------------- UI ----------------
    def _light_stylesheet(self) -> str:  # CSS for light theme
        return """
            QWidget { background:#f1c6b6; color:#1e1a18; }
            QPushButton { background:#a8937e; border-radius:6px; padding:8px 14px; color:#1e1a18; }
            QPushButton:disabled { background:#c9bbaf; color:#6e625a; }
            QLineEdit { background:#f6e5df; padding:6px; color:#1e1a18; }
            QTextEdit { background:#f6e5df; padding:10px; color:#1e1a18; }
            QLabel#title { font-size:22px; font-weight:bold; }
            QLabel.section { font-size:18px; font-weight:bold; }
            QLabel#instructions { background:#f6e5df; border:1px solid #caaea2; padding:10px; }
        """  # return a style sheet string

    def _dark_stylesheet(self) -> str:  # CSS for dark theme
        return """
            QWidget { background:#1f1f1f; color:#f0f0f0; }
            QPushButton { background:#3a3a3a; border-radius:6px; padding:8px 14px; color:#f0f0f0; }
            QPushButton:disabled { background:#2a2a2a; color:#777; }
            QLineEdit { background:#2b2b2b; padding:6px; color:#f0f0f0; border:1px solid #3d3d3d; }
            QTextEdit { background:#2b2b2b; padding:10px; color:#f0f0f0; border:1px solid #3d3d3d; }
            QLabel#title { font-size:22px; font-weight:bold; }
            QLabel.section { font-size:18px; font-weight:bold; }
            QLabel#instructions { background:#2b2b2b; border:1px solid #3d3d3d; padding:10px; }
        """  # return a style sheet string

    def _apply_theme(self):  # apply current theme stylesheet
        self.setStyleSheet(self._dark_stylesheet() if self.dark_mode else self._light_stylesheet())  # set stylesheet
        self.chk_dark.setChecked(self.dark_mode)  # sync checkbox state

    def _build_ui(self):  # create and lay out all UI widgets
        self.setWindowTitle(self.title)  # set main window title
        self.resize(1200, 800)  # initial size
        self.setMinimumSize(900, 600)  # minimum usable size

        root = QGridLayout(self)  # top-level grid layout
        root.setHorizontalSpacing(16)  # spacing between columns
        root.setVerticalSpacing(12)  # spacing between rows
        root.setContentsMargins(12, 12, 12, 12)  # outer margins

        # Top bar: Instructions + right-aligned toggles
        top_row = QHBoxLayout()  # horizontal layout for top bar
        instructions = QLabel(self)  # instructions label
        instructions.setObjectName("instructions")  # id for stylesheet
        instructions.setWordWrap(True)  # wrap long text
        instructions.setText(  # set instruction content
            "To use this software, load the obverse and the reverse of the coin you want to characterize. "
            "Then click Grade. FFT and Feature Match estimate wear on a 0–70 scale (0 = most worn, 70 = least)."
        )  # end text
        instructions.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  # expand horizontally
        top_row.addWidget(instructions, 10)  # add instructions taking most space

        # Dark mode toggle
        self.chk_dark = QCheckBox("Dark mode")  # checkbox to toggle dark theme
        self.chk_dark.stateChanged.connect(self._toggle_dark_mode)  # hook toggle handler
        top_row.addWidget(self.chk_dark, 0, Qt.AlignRight)  # place at right side

        root.addLayout(top_row, 0, 0, 1, 3)  # span top-row across all columns

        # Left image
        self.lbl_obverse = ImageLabel("./LincolnCent/Images/Obverse/Red/1928Penny65RD.jpg")  # default obverse image
        root.addWidget(self.lbl_obverse, 1, 0)  # place in row 1, col 0

        # Center: Status/Results panel
        center_col = QVBoxLayout()  # vertical layout for center column

        self.result_box = QTextEdit(self)  # central text area (status/results)
        self.result_box.setReadOnly(True)  # user cannot edit results
        self.result_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # grow to fill space
        center_col.addWidget(self.result_box, 1)  # add to center column

        # Busy indicator (hidden by default)
        busy_row = QHBoxLayout()  # horizontal layout for progress bar row
        busy_row.addStretch(1)  # push bar to center
        self.busy = QProgressBar()  # progress bar instance
        self.busy.setRange(0, 0)      # indeterminate spinner mode
        self.busy.setFixedWidth(220)  # consistent width
        self.busy.setVisible(False)  # hidden until grading
        busy_row.addWidget(self.busy)  # add progress bar
        busy_row.addStretch(1)  # keep centered
        center_col.addLayout(busy_row)  # add busy row under text

        root.addLayout(center_col, 1, 1)  # place center column at row 1 col 1

        # Right image
        self.lbl_reverse = ImageLabel("./LincolnCent/Images/Reverse/1928PennyBack65.jpg")  # default reverse image
        root.addWidget(self.lbl_reverse, 1, 2)  # place in row 1, col 2

        # File pickers row (obverse on left, reverse on right)
        left_controls = QHBoxLayout()  # left controls layout
        self.btn_load_obv = QPushButton("Load Obverse Of Coin")  # button to pick obverse
        self.btn_load_obv.clicked.connect(self.on_obverse_click)  # connect click handler
        self.txt_obv = QLineEdit(self)  # display chosen obverse path
        self.txt_obv.setPlaceholderText("Obverse filepath")  # hint text
        self.txt_obv.setReadOnly(True)  # not editable by user
        left_controls.addWidget(self.btn_load_obv, 2)  # add button with stretch factor
        left_controls.addWidget(self.txt_obv, 3)  # add path field with more space
        root.addLayout(left_controls, 2, 0)  # place in row 2, col 0

        # Spacer for the middle column (keeps grid tidy)
        root.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum), 2, 1)  # spacer in middle column

        # Right controls
        right_controls = QHBoxLayout()  # right controls layout
        self.btn_load_rev = QPushButton("Load Reverse Of Coin")  # button to pick reverse
        self.btn_load_rev.clicked.connect(self.on_reverse_click)  # connect click handler
        self.txt_rev = QLineEdit(self)  # display chosen reverse path
        self.txt_rev.setPlaceholderText("Reverse filepath")  # hint text
        self.txt_rev.setReadOnly(True)  # not editable
        right_controls.addWidget(self.btn_load_rev, 2)  # add button
        right_controls.addWidget(self.txt_rev, 3)  # add path field
        root.addLayout(right_controls, 2, 2)  # place in row 2, col 2

        # Bottom bar: Reset (left) + Grade (center)
        bottom_bar = QHBoxLayout()  # bottom bar layout
        self.btn_reset = QPushButton("Reset")  # reset button
        self.btn_reset.clicked.connect(self.on_reset_click)  # hook reset handler
        bottom_bar.addWidget(self.btn_reset, 0, Qt.AlignLeft)  # align left

        bottom_bar.addStretch(1)  # space before Grade

        self.btn_grade = QPushButton("Grade")  # grade button
        self.btn_grade.setMinimumWidth(240)  # make grade button prominent
        self.btn_grade.clicked.connect(self.on_grade_click)  # connect grade handler
        self.btn_grade.setEnabled(False)  # disabled until both images chosen
        bottom_bar.addWidget(self.btn_grade, 0, Qt.AlignCenter)  # place centered

        bottom_bar.addStretch(1)  # space after Grade
        root.addLayout(bottom_bar, 3, 0, 1, 3)  # span entire bottom row

        # Make central row/cols grow with window
        root.setRowStretch(1, 1)  # row 1 expands
        root.setColumnStretch(0, 1)  # left column expands
        root.setColumnStretch(1, 1)  # center column expands
        root.setColumnStretch(2, 1)  # right column expands

        # Initial status + theme
        self._apply_theme()  # apply stylesheet based on dark_mode
        self._update_status_panel()  # show initial instructions

        self.show()  # display the window

    # ---------- settings ----------
    def _restore_settings(self):  # load saved UI preferences
        # Dark mode
        self.dark_mode = bool(self.settings.value("dark_mode", False, type=bool))  # load theme flag
        self._apply_theme()  # apply theme right away

        # Window pos/size
        size = self.settings.value("win_size")  # read saved size
        pos = self.settings.value("win_pos")  # read saved position
        try:  # values may be missing or invalid
            if isinstance(size, QSize) and size.isValid():  # valid size?
                self.resize(size)  # restore size
            if isinstance(pos, QPoint):  # valid position?
                self.move(pos)  # restore position
        except Exception:  # corrupted settings
            pass  # ignore and keep defaults

    def _save_settings(self):  # persist current UI preferences
        self.settings.setValue("dark_mode", self.dark_mode)  # save theme flag
        self.settings.setValue("win_size", self.size())  # save window size
        self.settings.setValue("win_pos", self.pos())  # save window position

    # ---------- helpers ----------
    def _both_selected(self) -> bool:  # true if both image paths chosen
        return bool(self.obverse_path) and bool(self.reverse_path)  # check both non-empty

    def _status_text(self) -> str:  # build context-aware status text
        if not self.obverse_path and not self.reverse_path:  # nothing selected
            return (  # instruct user to pick both images
                "Next steps:\n\n"
                "• Select obverse image (left).\n"
                "• Select reverse image (right).\n"
                "• Then click Grade."
            )  # end text
        if not self.obverse_path:  # missing obverse
            return (  # prompt for obverse first
                "Next steps:\n\n"
                "• Select obverse image (left).\n"
                "• Then select reverse image and click Grade."
            )  # end text
        if not self.reverse_path:  # missing reverse
            return (  # prompt for reverse
                "Next steps:\n\n"
                "• Select reverse image (right).\n"
                "• Then click Grade."
            )  # end text
        return (  # both selected
            "Both images selected.\n\n"
            "Click Grade to compute: \n"
            "• FFT-based wear estimate\n"
            "• Feature-Match grade"
        )  # end text

    def _update_status_panel(self):  # refresh center panel and Grade button state
        self.result_box.setText(self._status_text())  # show current instructions/results
        self.btn_grade.setEnabled(self._both_selected() and not self.busy.isVisible())  # enable only when ready

    def _toggle_dark_mode(self, state: int):  # handle dark mode checkbox
        self.dark_mode = (state == Qt.Checked)  # store flag
        self._apply_theme()  # reapply stylesheet

    # ---------- file dialogs ----------
    def _start_dir(self, which: str) -> str:  # determine initial folder for dialog
        # which in {"obv", "rev"}
        key = "last_dir_obv" if which == "obv" else "last_dir_rev"  # pick settings key
        start = self.settings.value(key, "", type=str)  # read saved folder
        if start and Path(start).exists():  # if path still exists
            return start  # use saved folder
        return ""  # default to OS default

    def _store_dir(self, which: str, filepath: str):  # remember folder of selected file
        try:  # robust against odd paths
            p = Path(filepath).resolve().parent  # parent directory of file
            key = "last_dir_obv" if which == "obv" else "last_dir_rev"  # select key
            self.settings.setValue(key, str(p))  # save folder path
        except Exception:  # path issues
            pass  # ignore silently

    @pyqtSlot()  # Qt slot decorator
    def on_obverse_click(self):  # choose obverse image
        path, _ = QFileDialog.getOpenFileName(  # open file dialog
            self, "Select Obverse Image", self._start_dir("obv"),  # parent, title, start dir
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"  # filters
        )  # end dialog
        if not path:  # user canceled
            return  # no change
        self.obverse_path = path  # store selected path
        self.txt_obv.setText(path)  # show path in field
        self.lbl_obverse.set_image_path(path)  # preview image on left
        self._store_dir("obv", path)  # remember folder
        self._update_status_panel()  # refresh UI hints

    @pyqtSlot()  # Qt slot decorator
    def on_reverse_click(self):  # choose reverse image
        path, _ = QFileDialog.getOpenFileName(  # open file dialog
            self, "Select Reverse Image", self._start_dir("rev"),  # parent, title, start dir
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"  # filters
        )  # end dialog
        if not path:  # user canceled
            return  # no change
        self.reverse_path = path  # store selected path
        self.txt_rev.setText(path)  # show path in field
        self.lbl_reverse.set_image_path(path)  # preview image on right
        self._store_dir("rev", path)  # remember folder
        self._update_status_panel()  # refresh UI hints

    # ---------- Reset ----------
    @pyqtSlot()  # Qt slot decorator
    def on_reset_click(self):  # reset UI to defaults
        self.obverse_path = ""  # clear obverse path
        self.reverse_path = ""  # clear reverse path
        self.txt_obv.clear()  # clear obverse textbox
        self.txt_rev.clear()  # clear reverse textbox
        # Reset images back to defaults
        self.lbl_obverse.set_image_path("./LincolnCent/Images/Obverse/Red/1928Penny65RD.jpg")  # default obverse preview
        self.lbl_reverse.set_image_path("./LincolnCent/Images/Reverse/1928PennyBack65.jpg")  # default reverse preview
        # Reset results/status
        self.fft_results = "----"  # clear FFT results
        self.fm_results = "----"  # clear feature-match results
        self.busy.setVisible(False)  # hide spinner if visible
        self._update_status_panel()  # update texts and button states

    # ---------- Grade (threaded) ----------
    @pyqtSlot()  # Qt slot decorator
    def on_grade_click(self):  # start grading workflow
        if not self._both_selected() or self.busy.isVisible():  # guard invalid states
            self._update_status_panel()  # remind user what to do
            return  # abort click

        # show busy + lock controls
        self.busy.setVisible(True)  # show spinner
        self.btn_grade.setEnabled(False)  # disable grade button

        # spin up worker thread
        self.worker_thread = QThread(self)  # create background thread
        self.worker = GradeWorker(self.obverse_path, self.reverse_path)  # create worker with paths
        self.worker.moveToThread(self.worker_thread)  # move worker to thread

        self.worker_thread.started.connect(self.worker.run)  # run worker when thread starts
        self.worker.finished.connect(self._on_grade_finished)  # handle success
        self.worker.failed.connect(self._on_grade_failed)  # handle failure

        # teardown connections
        self.worker.finished.connect(self.worker_thread.quit)  # stop thread on finish
        self.worker.failed.connect(self.worker_thread.quit)  # stop thread on failure
        self.worker_thread.finished.connect(self.worker.deleteLater)  # delete worker safely
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)  # delete thread safely

        self.worker_thread.start()  # begin background grading

    @pyqtSlot(dict)  # expects dict payload
    def _on_grade_finished(self, data: dict):  # update UI on success
        self.fft_results = data.get("fft", "----")  # pull FFT text
        self.fm_results = data.get("fm", "----")  # pull FM text

        self.result_box.setText(  # show results in center panel
            f"FFT Analysis Results:\n{self.fft_results}\n\n"
            f"Feature Match Results:\n{self.fm_results}\n"
        )  # end text
        self.busy.setVisible(False)  # hide spinner
        self._update_status_panel()  # re-enable grade if allowed

    @pyqtSlot(str)  # expects string message
    def _on_grade_failed(self, message: str):  # show error if grading failed
        self.result_box.setText(f"An error occurred during grading:\n{message}")  # display error
        self.busy.setVisible(False)  # hide spinner
        self._update_status_panel()  # update buttons/status

    # ---------- lifecycle ----------
    def closeEvent(self, event):  # called when window closes
        self._save_settings()  # persist settings on exit
        super().closeEvent(event)  # continue default close


def runLWCCode():  # helper to run app from external caller
    app = QApplication(sys.argv)  # create Qt application
    ex = App()  # instantiate main window
    sys.exit(app.exec_())  # enter event loop


if __name__ == '__main__':  # run when executed as script
    app = QApplication(sys.argv)  # create Qt application
    ex = App()  # instantiate main window
    sys.exit(app.exec_())  # enter event loop
