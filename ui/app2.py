import sys
import numpy as np
import pickle
import pandas as pd
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QLabel, QScrollArea, QCheckBox, QPushButton, 
                           QLineEdit, QFormLayout, QGroupBox, QGridLayout,
                           QStackedWidget, QTextBrowser, QHBoxLayout, QComboBox,
                           QListWidget, QListWidgetItem, QFrame, QMessageBox)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect, QSize
from PyQt6.QtGui import QFont, QColor, QPalette

# --- Color Palettes ---
THEMES = {
    "Dark": {
        "sidebar": "#1e293b",
        "main_bg": "#0f172a",
        "card_bg": "#1e293b",
        "text": "#f8fafc",
        "text_muted": "#94a3b8",
        "accent": "#38bdf8",
        "accent_hover": "#0ea5e9",
        "border": "#334155",
        "input_bg": "#0f172a"
    },
    "Light": {
        "sidebar": "#ffffff",
        "main_bg": "#f8fafc",
        "card_bg": "#ffffff",
        "text": "#0f172a",
        "text_muted": "#64748b",
        "accent": "#0ea5e9",
        "accent_hover": "#0284c7",
        "border": "#e2e8f0",
        "input_bg": "#ffffff"
    }
}

class NeuralCareSymptom(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # State
        self.current_theme = "System"
        self.scale_pnt = 100
        self.sidebar_expanded = False
        self.sidebar_width = 175
        self.symptom_checks = {}
        self.model = None
        self.model_loaded = False
        
        # Window
        self.setWindowTitle('NeuralCare-Symptom')
        self.setMinimumSize(1000, 800)
        
        self.load_model_and_data()
        
        # Root Layout: [Header] (Row 1) -> [Body] (Row 2)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.root_layout = QVBoxLayout(self.central_widget)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)
        
        # --- ROW 1: Header ---
        self.setup_header()
        
        # --- ROW 2: Body (Sidebar + Content) ---
        self.body_widget = QWidget()
        self.body_layout = QHBoxLayout(self.body_widget)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(0)
        
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(0)
        self.setup_sidebar_content()
        
        self.stack = QStackedWidget()
        self.stack.addWidget(self.create_form_page())
        self.stack.addWidget(self.create_results_page())
        self.stack.currentChanged.connect(self.on_page_changed)
        
        self.body_layout.addWidget(self.sidebar)
        self.body_layout.addWidget(self.stack, 1)
        
        self.root_layout.addWidget(self.body_widget, 1)
        
        # Animations
        self.sidebar_anim = QPropertyAnimation(self.sidebar, b"minimumWidth")
        self.sidebar_anim.setDuration(200)
        self.sidebar_anim.setEasingCurve(QEasingCurve.Type.InOutSine)

        self.apply_theme()

    def setup_header(self):
        header = QFrame()
        header.setFixedHeight(70)
        header.setObjectName("Header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)
        
        self.menu_btn = QPushButton("☰")
        self.menu_btn.setObjectName("MenuBtn")
        self.menu_btn.setFixedSize(40, 40)
        self.menu_btn.clicked.connect(self.toggle_sidebar)
        layout.addWidget(self.menu_btn)
        
        layout.addSpacing(15)
        self.title_label = QLabel("NeuralCare-Symptom")
        self.title_label.setObjectName("MainTitle")
        layout.addWidget(self.title_label)

        layout.addStretch()

        self.submit_btn = QPushButton("ANALYZE NOW")
        self.submit_btn.setObjectName("ActionBtn")
        self.submit_btn.clicked.connect(self.submit_form)
        layout.addWidget(self.submit_btn)

        self.root_layout.addWidget(header)

    def setup_sidebar_content(self):
        vbox = QVBoxLayout(self.sidebar)
        vbox.setContentsMargins(15, 20, 15, 20)
        vbox.setSpacing(10)

        # Theme
        vbox.addWidget(QLabel("THEME", objectName="SubLabel"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "System"])
        self.theme_combo.setMaxVisibleItems(3)
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        vbox.addWidget(self.theme_combo)

        vbox.addSpacing(15)

        # Scaling
        vbox.addWidget(QLabel("SCALE", objectName="SubLabel"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["80%", "100%", "120%", "150%", "200%"])
        self.scale_combo.setCurrentIndex(1)
        self.scale_combo.currentTextChanged.connect(self.on_scale_changed)
        vbox.addWidget(self.scale_combo)

        vbox.addStretch()

    def apply_theme(self):
        p = THEMES["Dark" if self.current_theme == "System" else self.current_theme]
        # Safeguard font size to be at least 8pt
        fs = max(8, int(14 * (self.scale_pnt / 100)))
        
        qss = f"""
        * {{ font-family: 'Segoe UI', system-ui; }}
        
        QMainWindow {{ background-color: {p['main_bg']}; }}
        
        QFrame#Header {{ 
            background-color: {p['main_bg']}; 
            border-bottom: 1px solid {p['border']};
        }}
        
        QFrame#Sidebar {{
            background-color: {p['sidebar']};
            border-bottom-right-radius: 20px;
            border-right: 1px solid {p['border']};
        }}
        
        QLabel#MainTitle {{ color: {p['accent']}; font-size: {fs + 10}px; font-weight: 800; }}
        QLabel#SubLabel {{ color: {p['text_muted']}; font-size: {fs + 1}px; font-weight: bold; padding-top: 10px; }}
        QLabel {{ color: {p['text']}; font-size: {fs}px; }}

        QPushButton#MenuBtn {{
            background-color: transparent;
            color: {p['text']};
            border: 1px solid {p['border']};
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
        }}
        QPushButton#MenuBtn:hover {{ background-color: {p['border']}; }}

        QPushButton#ActionBtn {{
            background-color: {p['accent']};
            color: #ffffff;
            border: none;
            border-radius: 10px;
            padding: 8px 20px;
            font-weight: 700;
            font-size: {fs}px;
        }}
        QPushButton#ActionBtn:hover {{ background-color: {p['accent_hover']}; }}

        QLineEdit, QComboBox, QListWidget, QTextBrowser {{
            background-color: {p['input_bg']};
            border: 1px solid {p['border']};
            border-radius: 8px;
            color: {p['text']};
            padding: 8px;
            font-size: {fs}px;
        }}
        
        QListWidget {{
            outline: none;
            background-color: {p['input_bg']};
            border: 1px solid {p['border']};
        }}
        
        QListWidget::item {{ 
            background-color: {p['sidebar']};
            color: {p['text']};
            padding: 10px; 
            margin: 4px 8px; 
            border-radius: 8px;
            border: 1px solid {p['border']};
        }}
        
        QListWidget::item:hover {{
            background-color: {p['border']};
        }}
        
        QListWidget::item:selected {{
            background-color: {p['accent']};
            color: #ffffff;
            border-color: {p['accent']};
        }}

        QGroupBox {{
            border: 1px solid {p['border']};
            border-radius: 15px;
            background-color: {p['card_bg']};
            margin-top: 10px;
            padding-top: 30px;
            color: {p['text']};
            font-weight: bold;
            font-size: {fs + 6}px;
        }}

        QCheckBox {{ color: {p['text']}; spacing: 10px; }}
        QCheckBox::indicator {{ 
            width: 20px; height: 20px; 
            border: 2px solid {p['border']}; 
            border-radius: 5px; 
            background-color: {p['main_bg']};
        }}
        
        QCheckBox::indicator:checked {{ 
            background-color: {p['accent']}; 
            border-color: {p['accent']};
        }}
        
        QCheckBox::indicator:unchecked:hover {{
            border-color: {p['accent']};
        }}

        QScrollArea, QScrollArea > QWidget, QWidget#ScrollContent {{
            background-color: transparent;
            border: none;
        }}

        QFrame#DiagnosisCard {{
            background-color: {p['accent']}15; /* 15 is ~8% opacity in hex */
            border: 2px solid {p['accent']};
            border-radius: 12px;
        }}
        """
        self.setStyleSheet(qss)

        # Override palette as a secondary measure for standard widgets
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.WindowText, QColor(p['text']))
        pal.setColor(QPalette.ColorRole.Base, QColor(p['main_bg']))
        pal.setColor(QPalette.ColorRole.Text, QColor(p['text']))
        pal.setColor(QPalette.ColorRole.Highlight, QColor(p['accent']))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        self.setPalette(pal)

        # Update results card style if it exists
        if hasattr(self, 'res_card_frame'):
            self.res_card_frame.setStyleSheet(f"background-color: {p['accent']}15; border: 2px solid {p['accent']}; border-radius: 12px;")

        self.sidebar.setFixedWidth(self.sidebar_width if self.sidebar_expanded else 0)
        # Ensure sidebar is visible if expanded
        if self.sidebar_expanded:
            self.sidebar.setMinimumWidth(self.sidebar_width)
        else:
            self.sidebar.setMinimumWidth(0)

    def toggle_sidebar(self):
        if self.sidebar_expanded:
            self.menu_btn.setText("☰")
            self.sidebar_anim.setStartValue(self.sidebar_width)
            self.sidebar_anim.setEndValue(0)
            self.sidebar_expanded = False
        else:
            self.menu_btn.setText("✕")
            self.sidebar_anim.setStartValue(0)
            self.sidebar_anim.setEndValue(self.sidebar_width)
            self.sidebar_expanded = True
        self.sidebar_anim.start()

    def on_theme_changed(self, val):
        self.current_theme = val
        self.apply_theme()

    def on_scale_changed(self, val):
        self.scale_pnt = int(val.replace('%', ''))
        self.apply_theme()

    def create_form_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 20, 30, 30)

        info_box = QGroupBox("Patient Profile")
        grid = QGridLayout(info_box)
        grid.setSpacing(10)

        self.name_in = QLineEdit()
        self.name_in.setPlaceholderText("Full Name (Optional)")
        grid.addWidget(QLabel("NAME", objectName="SubLabel"), 0, 0)
        grid.addWidget(self.name_in, 1, 0)

        self.age_in = QLineEdit()
        self.age_in.setPlaceholderText("Enter Age (1-120)")
        # self.age_in.setFixedWidth(200)
        grid.addWidget(QLabel("AGE *", objectName="SubLabel"), 0, 1)
        grid.addWidget(self.age_in, 1, 1)

        self.gen_in = QComboBox()
        self.gen_in.addItems(["Male", "Female", "Other"])
        self.gen_in.setFixedWidth(250)
        grid.addWidget(QLabel("GENDER *", objectName="SubLabel"), 0, 2)
        grid.addWidget(self.gen_in, 1, 2)
        
        # Height and Weight in new row
        self.height_in = QLineEdit()
        self.height_in.setPlaceholderText("Height")
        # self.height_in.setFixedWidth(120)
        grid.addWidget(QLabel("HEIGHT (cm) *", objectName="SubLabel"), 2, 0)
        grid.addWidget(self.height_in, 3, 0)
        
        self.weight_in = QLineEdit()
        self.weight_in.setPlaceholderText("Weight")
        # self.weight_in.setFixedWidth(120)
        grid.addWidget(QLabel("WEIGHT (kg) *", objectName="SubLabel"), 2, 1)
        grid.addWidget(self.weight_in, 3, 1)

        self.bmi_in = QLineEdit()
        self.bmi_in.setPlaceholderText("BMI Score")
        self.bmi_in.setReadOnly(True)
        self.bmi_in.setFixedWidth(250)
        grid.addWidget(QLabel("BODY MASS INDEX (BMI)", objectName="SubLabel"), 2, 2)
        grid.addWidget(self.bmi_in, 3, 2)

        # Connections for BMI
        self.height_in.textChanged.connect(self.calculate_bmi)
        self.weight_in.textChanged.connect(self.calculate_bmi)
        
        layout.addWidget(info_box)
        
        # Row 2: Symptom Split
        sym_box = QGroupBox("Symptoms")
        sym_layout = QHBoxLayout(sym_box)
        
        # Left Side (Search + Scroll)
        left_v = QVBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("🔍 Search symptoms...")
        self.search.textChanged.connect(self.filter_syms)
        left_v.addWidget(self.search)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("ScrollContent")
        self.g_layout = QGridLayout(self.scroll_content)
        
        sorted_keys = sorted(self.symptoms_dict.keys())
        for i, s in enumerate(sorted_keys):
            cb = QCheckBox(s.replace('_', ' ').title())
            cb.setObjectName(s)
            cb.stateChanged.connect(self.on_check)
            self.symptom_checks[s] = cb
            self.g_layout.addWidget(cb, i // 2, i % 2)
        
        scroll.setWidget(self.scroll_content)
        left_v.addWidget(scroll)
        
        sym_layout.addLayout(left_v, 2)
        
        # Right Side (Selection)
        right_v = QVBoxLayout()
        right_v.addWidget(QLabel("SELECTED SYMPTOMS", objectName="SubLabel"))
        self.sel_list = QListWidget()
        self.sel_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        # Tight spacing for list
        self.sel_list.setStyleSheet("QListWidget::item { padding: 1px; }")
        right_v.addWidget(self.sel_list)
        
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.setObjectName("SecondaryBtn")
        self.remove_btn.clicked.connect(self.remove_selected_symptoms)
        right_v.addWidget(self.remove_btn)
        
        sym_layout.addLayout(right_v, 1)
        
        layout.addWidget(sym_box)
        
        return page

    def create_results_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 20, 30, 30)
        
        self.res_title = QLabel("PREDICTED CLINICAL REPORT")
        self.res_title.setStyleSheet("font-size: 20px; font-weight: 800; color: #38bdf8; margin-bottom: 5px;")
        layout.addWidget(self.res_title)

        # Header Section: 2 Columns
        head_row = QHBoxLayout()
        
        # Col 1 Containers (Split into Rows)
        col1_v = QVBoxLayout()
        self.info_group = QGroupBox("Patient Data")
        self.info_group.setFixedWidth(275) # Fixed width
        self.info_v = QVBoxLayout(self.info_group)
        self.res_patient_data = QLabel()
        self.res_patient_data.setStyleSheet("line-height: 150%;")
        self.info_v.addWidget(self.res_patient_data)
        col1_v.addWidget(self.info_group, 3) # Majority height

        self.reeval_btn = QPushButton("EDIT ASSESSMENT")
        self.reeval_btn.setObjectName("ActionBtn") # Primary style
        self.reeval_btn.setFixedHeight(45)
        self.reeval_btn.setFixedWidth(275)
        self.reeval_btn.clicked.connect(self.go_back_to_edit)
        col1_v.addWidget(self.reeval_btn, 1) # Fixed height row
        
        head_row.addLayout(col1_v)

        # Col 2: Symptoms
        self.sym_group = QGroupBox("Analyzed Symptoms")
        sym_h = QVBoxLayout(self.sym_group)
        self.res_symptoms_data = QLabel()
        self.res_symptoms_data.setStyleSheet("font-size: 16px; padding: 10px;")
        self.res_symptoms_data.setWordWrap(True)
        sym_h.addWidget(self.res_symptoms_data)
        head_row.addWidget(self.sym_group, 1)
        
        layout.addLayout(head_row)
        
        card = QFrame()
        card.setObjectName("DiagnosisCard")
        v_card = QVBoxLayout(card)
        self.main_diag = QLabel("Diagnosis Loading...")
        self.main_diag.setStyleSheet("font-size: 28px; font-weight: 800; padding-top: 10px;")
        v_card.addWidget(self.main_diag)
        
        # Integrated Description
        self.res_diag_desc = QLabel()
        self.res_diag_desc.setWordWrap(True)
        self.res_diag_desc.setStyleSheet("color: #94a3b8; font-size: 16px; line-height: 140%; padding-bottom: 10px; padding-left: 10px;")
        v_card.addWidget(self.res_diag_desc)
        layout.addWidget(card)
        
        grid = QGridLayout()
        grid.setSpacing(10)
        self.b_prec = QTextBrowser()
        self.b_meds = QTextBrowser()
        self.b_diet = QTextBrowser()
        self.b_work = QTextBrowser()
        
        grid.addWidget(QLabel("ACTIONABLE PRECAUTIONS", objectName="SubLabel"), 0, 0)
        grid.addWidget(self.b_prec, 1, 0)
        grid.addWidget(QLabel("MANAGEMENT MEDICATIONS", objectName="SubLabel"), 0, 1)
        grid.addWidget(self.b_meds, 1, 1)
        grid.addWidget(QLabel("NUTRITIONAL GUIDELINES", objectName="SubLabel"), 2, 0)
        grid.addWidget(self.b_diet, 3, 0)
        grid.addWidget(QLabel("LIFESTYLE & WORKOUT", objectName="SubLabel"), 2, 1)
        grid.addWidget(self.b_work, 3, 1)
        
        layout.addLayout(grid)
        return page

    def filter_syms(self, txt):
        q = txt.lower()
        for k, cb in self.symptom_checks.items():
            cb.setVisible(q in k.replace('_', ' ').lower())

    def on_check(self, state):
        cb = self.sender()
        if state == 2:
            # Check if already in list to avoid duplicates
            items = self.sel_list.findItems(cb.text(), Qt.MatchFlag.MatchExactly)
            if not items:
                item = QListWidgetItem(cb.text())
                item.setData(Qt.ItemDataRole.UserRole, cb.objectName())
                self.sel_list.addItem(item)
        else:
            items = self.sel_list.findItems(cb.text(), Qt.MatchFlag.MatchExactly)
            for i in items: self.sel_list.takeItem(self.sel_list.row(i))

    def remove_selected_symptoms(self):
        """Removes selected items from list and unchecks corresponding checkboxes"""
        selected_items = self.sel_list.selectedItems()
        if not selected_items:
            return
            
        for item in selected_items:
            symptom_key = item.data(Qt.ItemDataRole.UserRole)
            if symptom_key in self.symptom_checks:
                # Unchecking the checkbox will trigger on_check which removes the item from the list
                # However, to avoid issues during loop, we block signals or handle it carefully
                self.symptom_checks[symptom_key].setChecked(False)
            else:
                # Fallback if somehow key isn't found
                self.sel_list.takeItem(self.sel_list.row(item))

    def validate(self):
        # Age, Gender, Height, Weight compulsory
        age = self.age_in.text().strip()
        h = self.height_in.text().strip()
        w = self.weight_in.text().strip()
        
        try:
            if not age or int(age) <= 0 or int(age) > 120: raise ValueError
        except:
            QMessageBox.warning(self, "Invalid Entry", "Age must be within 1-120 years.")
            return False
            
        try:
            if not h or float(h) < 30 or float(h) > 250: raise ValueError
        except:
            QMessageBox.warning(self, "Invalid Entry", "Physical height must be within 30-250 cm.")
            return False
            
        try:
            if not w or float(w) < 1 or float(w) > 500: raise ValueError
        except:
            QMessageBox.warning(self, "Invalid Entry", "Physical weight must be within 1-500 kg.")
            return False
            
        if self.sel_list.count() == 0:
            QMessageBox.warning(self, "No Symptoms", "Select symptoms to analyze progress.")
            return False
        return True

    def submit_form(self):
        if self.stack.currentIndex() == 1:
            self.reset_app()
            return
            
        if not self.validate(): return
        
        # Collect info
        name = self.name_in.text().strip()
        age = self.age_in.text().strip()
        gen = self.gen_in.currentText()
        h = self.height_in.text().strip()
        w = self.weight_in.text().strip()
        bmi = self.bmi_in.text()
        
        # Build Patient Detail string
        info_lines = []
        if name: info_lines.append(f"<b>Name:</b> {name}")
        info_lines.append(f"<b>Age: </b> {age} Years")
        info_lines.append(f"<b>Gender: </b> {gen}")
        info_lines.append(f"<b>Physicals: </b> {h}cm / {w}kg")
        info_lines.append(f"<b>BMI: </b> {bmi}")
        self.res_patient_data.setText("<br>".join(info_lines))

        # Build Symptoms string
        selected_texts = [self.sel_list.item(i).text() for i in range(self.sel_list.count())]
        self.res_symptoms_data.setText(" • " + " • ".join(selected_texts))

        syms = [self.sel_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.sel_list.count())]
        v = np.zeros(len(self.symptoms_dict))
        for s in syms:
            if s in self.symptoms_dict: v[self.symptoms_dict[s]] = 1
        
        df = pd.DataFrame([v], columns=list(self.symptoms_dict.keys()))
        
        if not self.model:
            QMessageBox.critical(self, "Model Error", "The AI model could not be loaded. Please check your installation.")
            return

        res = self.diseases_list.get(self.model.predict(df)[0], "Unknown")
        
        # Display data
        self.main_diag.setText(res)
        self.res_diag_desc.setText(self.get_data(self.sym_des, res, 'Description'))
        
        self.b_prec.setHtml(self.get_data_list(self.precautions, res, ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']))
        self.b_meds.setHtml(self.get_data_list(self.medications, res, ['Medication']))
        self.b_diet.setHtml(self.get_data_list(self.diets, res, ['Diet']))
        self.b_work.setHtml(self.get_data_list(self.workout, res, ['workout']))
        
        self.stack.setCurrentIndex(1)

    def on_page_changed(self, index):
        if index == 0:
            self.submit_btn.setText("ANALYZE NOW")
        else:
            self.submit_btn.setText("NEW ANALYSIS")

    def go_back_to_edit(self):
        self.stack.setCurrentIndex(0)

    def get_data(self, df, disease, col):
        try: return df[df['Disease' if 'Disease' in df.columns else 'disease'] == disease][col].iloc[0]
        except: return "Detailed data unavailable."

    def get_data_list(self, df, disease, cols):
        try:
            row = df[df['Disease' if 'Disease' in df.columns else 'disease'] == disease][cols].values[0]
            items = []
            for raw_val in row:
                val = str(raw_val)
                if val.lower() == 'nan' or not val.strip():
                    continue
                
                # Handle string-represented lists e.g. "['a', 'b']"
                if val.startswith('[') and val.endswith(']'):
                    try:
                        import ast
                        actual_list = ast.literal_eval(val)
                        if isinstance(actual_list, list):
                            items.extend([str(x) for x in actual_list])
                            continue
                    except:
                        pass
                
                items.append(val)
                
            if not items: return "<ul><li>N/A</li></ul>"
            return "<ul>" + "".join([f"<li>{x}</li>" for x in items]) + "</ul>"
        except: return "<ul><li>N/A</li></ul>"

    def calculate_bmi(self):
        """Calculates BMI and updates display with status"""
        try:
            h_text = self.height_in.text().strip()
            w_text = self.weight_in.text().strip()
            
            if not h_text or not w_text:
                self.bmi_in.clear()
                self.bmi_in.setStyleSheet("")
                return

            h = float(h_text)
            w = float(w_text)
            
            if h > 0 and w > 0:
                bmi = w / ((h / 100) ** 2)
                
                status = ""
                color = ""
                if bmi < 18.5:
                    status = "Underweight"
                    color = "#fbbf24" # Amber
                elif 18.5 <= bmi < 25:
                    status = "Normal"
                    color = "#22c55e" # Green
                elif 25 <= bmi < 30:
                    status = "Overweight"
                    color = "#f97316" # Orange
                else:
                    status = "Obese"
                    color = "#ef4444" # Red
                
                self.bmi_in.setText(f"{bmi:.1f} ({status})")
                self.bmi_in.setStyleSheet(f"color: {color}; font-weight: bold; border-color: {color};")
            else:
                self.bmi_in.clear()
                self.bmi_in.setStyleSheet("")
        except ValueError:
            self.bmi_in.setText("Invalid Input")
            self.bmi_in.setStyleSheet("color: #ef4444;")

    def reset_app(self):
        self.age_in.clear()
        self.height_in.clear()
        self.weight_in.clear()
        self.name_in.clear()
        self.sel_list.clear()
        for c in self.symptom_checks.values(): c.setChecked(False)
        self.stack.setCurrentIndex(0)
        self.submit_btn.setEnabled(True)

    def load_model_and_data(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_path, "..", "data")
        model_path = os.path.join(base_path, "..", "model.pkl")

        self.symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
        self.diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

        try:
            self.sym_des = pd.read_csv(os.path.join(data_path, "description.csv"))
            self.precautions = pd.read_csv(os.path.join(data_path, "precautions_df.csv"))
            self.medications = pd.read_csv(os.path.join(data_path, "medications.csv"))
            self.diets = pd.read_csv(os.path.join(data_path, "diets.csv"))
            self.workout = pd.read_csv(os.path.join(data_path, "workout_df.csv"))
        except: pass

        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            self.model_loaded = False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = NeuralCareSymptom()
    window.show()
    sys.exit(app.exec())
