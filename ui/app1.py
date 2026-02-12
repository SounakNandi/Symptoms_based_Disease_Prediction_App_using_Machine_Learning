import sys
import numpy as np
import pickle
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QLabel, QScrollArea, QCheckBox, QPushButton, 
                           QLineEdit, QFormLayout, QGroupBox, QGridLayout,
                           QStackedWidget, QTextBrowser, QHBoxLayout, QComboBox,
                           QListWidget, QListWidgetItem)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class MedicalDiagnosisSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Get screen size and set window size to half of it
        screen_size = QApplication.primaryScreen().size()
        self.width = screen_size.width() // 2
        self.height = screen_size.height() // 2
        
        # Load ML model and related data
        self.load_model_and_data()
        
        # Create stacked widget to handle multiple pages
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Create the form page and results page
        self.form_page = self.create_form_page()
        self.results_page = self.create_results_page()
        
        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.form_page)
        self.stacked_widget.addWidget(self.results_page)

        # Create checkboxes for symptoms
        self.symptom_checks = {}

        # Initialize the UI
        self.initUI()
        
    def load_model_and_data(self):
        """Load the ML model and related data"""
        # Define symptoms dictionary and diseases list (always available)
        self.symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
        self.diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

        # Load datasets
        try:
            self.sym_des = pd.read_csv("../data/symtoms_df.csv")
            self.precautions = pd.read_csv("../data/precautions_df.csv")
            self.workout = pd.read_csv("../data/workout_df.csv")
            self.description = pd.read_csv("../data/description.csv")
            self.medications = pd.read_csv("../data/medications.csv")
            self.diets = pd.read_csv("../data/diets.csv")
        except Exception as e:
            print(f"Error loading CSV data: {e}")

        # Load the ML model
        try:
            with open('../model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False

    
    def initUI(self):
        # Set window title and size
        self.setWindowTitle('Medical Diagnosis System')
        self.setGeometry(0, 0, self.width, self.height)
        
        # Center the window on the screen
        self.center()
        
    def create_form_page(self):
        # Create form page widget
        form_page = QWidget()
        main_layout = QVBoxLayout(form_page)
        
        # Add title label
        title_label = QLabel("Medical Diagnosis System")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create form for personal details
        personal_details = QGroupBox("Personal Information")
        form_layout = QFormLayout()
        
        # Add form fields
        self.name_input = QLineEdit()
        self.age_input = QLineEdit()
        self.gender_input = QComboBox()
        self.gender_input.addItems(["Male", "Female", "Other"])
        
        form_layout.addRow("Full Name:", self.name_input)
        form_layout.addRow("Age:", self.age_input)
        form_layout.addRow("Gender:", self.gender_input)
        
        personal_details.setLayout(form_layout)
        main_layout.addWidget(personal_details)
        
        # Create symptoms selection section
        symptoms_group = QGroupBox("Select Your Symptoms")
        symptoms_layout = QVBoxLayout()
        
        # Add search box
        self.symptom_search = QLineEdit()
        self.symptom_search.setPlaceholderText("Search for symptoms...")
        self.symptom_search.textChanged.connect(self.filter_symptoms)
        symptoms_layout.addWidget(self.symptom_search)
        
        # Create list for selected symptoms
        self.selected_symptoms_list = QListWidget()
        symptoms_layout.addWidget(QLabel("Selected Symptoms"))
        symptoms_layout.addWidget(self.selected_symptoms_list)
        
        # Add button to remove selected symptom
        remove_button = QPushButton("Remove Selected Symptom")
        remove_button.clicked.connect(self.remove_selected_symptom)
        symptoms_layout.addWidget(remove_button)
        
        # Create scroll area for symptoms
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Create widget to hold checkboxes for symptoms
        symptoms_widget = QWidget()
        self.symptoms_grid = QGridLayout(symptoms_widget)
        
        # Get all symptoms
        if hasattr(self, 'symptoms_dict'):
            symptom_list = list(self.symptoms_dict.keys())
            symptom_list.sort()

            # Create checkboxes for symptoms
            self.symptom_checks = {}

            # Arrange checkboxes in a grid (3 columns)
            for i, symptom in enumerate(symptom_list):
                checkbox = QCheckBox(symptom.replace('_', ' ').title())
                checkbox.setObjectName(symptom)
                checkbox.stateChanged.connect(self.symptom_checked)
                self.symptom_checks[symptom] = checkbox
                row, col = i // 3, i % 3
                self.symptoms_grid.addWidget(checkbox, row, col)
        
        scroll_area.setWidget(symptoms_widget)
        symptoms_layout.addWidget(scroll_area)
        symptoms_group.setLayout(symptoms_layout)
        
        main_layout.addWidget(symptoms_group)
        
        # Add submit button
        submit_button = QPushButton("Submit for Diagnosis")
        submit_button.clicked.connect(self.submit_form)
        main_layout.addWidget(submit_button)
        
        # Add disclaimer
        disclaimer = QLabel("DISCLAIMER: This application is for informational purposes only. It does not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.")
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet("color: red; font-weight: bold;")
        main_layout.addWidget(disclaimer)
        
        return form_page
    
    def create_results_page(self):
        # Create results page widget
        results_page = QWidget()
        results_layout = QVBoxLayout(results_page)
        
        # Add title
        results_title = QLabel("Diagnosis Results")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        results_title.setFont(title_font)
        results_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(results_title)
        
        # Patient info section
        self.patient_info = QLabel()
        self.patient_info.setFrameStyle(QLabel.Shape.Panel | QLabel.Shadow.Sunken)
        self.patient_info.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.patient_info.setWordWrap(True)
        results_layout.addWidget(self.patient_info)
        
        # Create sections for diagnosis information
        # Diagnosis Section
        diagnosis_group = QGroupBox("Predicted Condition")
        diagnosis_layout = QVBoxLayout()
        self.diagnosis_text = QTextBrowser()
        diagnosis_layout.addWidget(self.diagnosis_text)
        diagnosis_group.setLayout(diagnosis_layout)
        results_layout.addWidget(diagnosis_group)
        
        # Description Section
        description_group = QGroupBox("Description")
        description_layout = QVBoxLayout()
        self.description_text = QTextBrowser()
        description_layout.addWidget(self.description_text)
        description_group.setLayout(description_layout)
        results_layout.addWidget(description_group)
        
        # Precautions Section
        precautions_group = QGroupBox("Recommended Precautions")
        precautions_layout = QVBoxLayout()
        self.precautions_text = QTextBrowser()
        precautions_layout.addWidget(self.precautions_text)
        precautions_group.setLayout(precautions_layout)
        results_layout.addWidget(precautions_group)
        
        # Medications Section
        medications_group = QGroupBox("Recommended Medications")
        medications_layout = QVBoxLayout()
        self.medications_text = QTextBrowser()
        medications_layout.addWidget(self.medications_text)
        medications_group.setLayout(medications_layout)
        results_layout.addWidget(medications_group)
        
        # Diet Section
        diet_group = QGroupBox("Recommended Diet")
        diet_layout = QVBoxLayout()
        self.diet_text = QTextBrowser()
        diet_layout.addWidget(self.diet_text)
        diet_group.setLayout(diet_layout)
        results_layout.addWidget(diet_group)
        
        # Workout Section
        workout_group = QGroupBox("Recommended Workout")
        workout_layout = QVBoxLayout()
        self.workout_text = QTextBrowser()
        workout_layout.addWidget(self.workout_text)
        workout_group.setLayout(workout_layout)
        results_layout.addWidget(workout_group)
        
        # Add disclaimer
        disclaimer = QLabel("DISCLAIMER: This application is for informational purposes only. It does not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.")
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet("color: red; font-weight: bold;")
        results_layout.addWidget(disclaimer)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        back_button = QPushButton("Back to Form")
        back_button.clicked.connect(self.go_back_to_form)
        new_diagnosis_button = QPushButton("New Diagnosis")
        new_diagnosis_button.clicked.connect(self.clear_and_reset)
        
        button_layout.addWidget(back_button)
        button_layout.addWidget(new_diagnosis_button)
        results_layout.addLayout(button_layout)
        
        return results_page
    
    def center(self):
        """Center the window on the screen"""
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )
    
    def symptom_checked(self, state):
        """Handler for when a symptom checkbox is checked/unchecked"""
        checkbox = self.sender()
        symptom = checkbox.objectName()
        
        if state == Qt.CheckState.Checked.value:
            # Add to the selected symptoms list
            item = QListWidgetItem(symptom.replace('_', ' ').title())
            item.setData(Qt.ItemDataRole.UserRole, symptom)
            self.selected_symptoms_list.addItem(item)
        else:
            # Remove from the selected symptoms list
            items = self.selected_symptoms_list.findItems(symptom.replace('_', ' ').title(), Qt.MatchFlag.MatchExactly)
            for item in items:
                row = self.selected_symptoms_list.row(item)
                self.selected_symptoms_list.takeItem(row)
    
    def remove_selected_symptom(self):
        """Remove a symptom from the selected list"""
        current_item = self.selected_symptoms_list.currentItem()
        if current_item:
            symptom = current_item.data(Qt.ItemDataRole.UserRole)
            # Uncheck the checkbox
            if symptom in self.symptom_checks:
                self.symptom_checks[symptom].setChecked(False)
            # Remove from the list
            row = self.selected_symptoms_list.row(current_item)
            self.selected_symptoms_list.takeItem(row)
    
    def filter_symptoms(self, text):
        """Filter symptoms based on search text"""
        search_text = text.lower()
        for symptom, checkbox in self.symptom_checks.items():
            if search_text in symptom.lower():
                checkbox.setVisible(True)
            else:
                checkbox.setVisible(False)
    
    def get_helper_data(self, disease):
        """Get detailed information about the disease"""
        try:
            # Get description
            desc = self.description[self.description['Disease'] == disease]['Description']
            desc = " ".join([w for w in desc]) if not desc.empty else "No description available."
            
            # Get precautions
            pre = self.precautions[self.precautions['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
            pre = [col for col in pre.values] if not pre.empty else [["No precautions available."]]
            
            # Get medications
            med = self.medications[self.medications['Disease'] == disease]['Medication']
            med = [m for m in med.values] if not med.empty else ["No medication information available."]
            
            # Get diet recommendations
            die = self.diets[self.diets['Disease'] == disease]['Diet']
            die = [d for d in die.values] if not die.empty else ["No diet information available."]
            
            # Get workout recommendations
            wrkout = self.workout[self.workout['disease'] == disease]['workout']
            wrkout = [w for w in wrkout.values] if not wrkout.empty else ["No workout information available."]
            
            return desc, pre, med, die, wrkout
        except Exception as e:
            print(f"Error getting helper data: {e}")
            return ("No information available.", [["No precautions available."]], 
                   ["No medication information available."], 
                   ["No diet information available."], 
                   ["No workout information available."])
    
    def get_predicted_disease(self, patient_symptoms):
        """Predict disease based on symptoms"""
        if not self.model_loaded:
            return "Error: AI Model could not be loaded due to environment compatibility issues. Please check the terminal for details."
        
        try:
            # Create an input vector based on selected symptoms
            input_vector = np.zeros(len(self.symptoms_dict))
            for symptom in patient_symptoms:
                if symptom in self.symptoms_dict:
                    input_vector[self.symptoms_dict[symptom]] = 1
            
            # Make a prediction - Wrap in a DataFrame to avoid feature name warnings
            input_df = pd.DataFrame([input_vector], columns=list(self.symptoms_dict.keys()))
            prediction = self.model.predict(input_df)[0]
            return self.diseases_list[prediction]
        except Exception as e:
            print(f"Error making prediction: {e}")
            return "Unable to make a prediction. Please try again."
    
    def submit_form(self):
        """Process the form submission and switch to results page"""
        # Validate inputs
        name = self.name_input.text().strip()
        age = self.age_input.text().strip()
        gender = self.gender_input.currentText()
        
        if not name or not age:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Missing Information", "Please provide your name and age.")
            return
        
        # Get selected symptoms
        selected_symptoms = []
        for i in range(self.selected_symptoms_list.count()):
            item = self.selected_symptoms_list.item(i)
            selected_symptoms.append(item.data(Qt.ItemDataRole.UserRole))
        
        if not selected_symptoms:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Symptoms Selected", "Please select at least one symptom.")
            return
        
        # Predict disease
        predicted_disease = self.get_predicted_disease(selected_symptoms)
        
        # Get additional information
        desc, precautions, medications, diet, workout = self.get_helper_data(predicted_disease)
        
        # Display patient info
        formatted_symptoms = [s.replace('_', ' ').title() for s in selected_symptoms]
        patient_info_text = f"<b>Patient:</b> {name}<br>"
        patient_info_text += f"<b>Age:</b> {age}<br>"
        patient_info_text += f"<b>Gender:</b> {gender}<br>"
        patient_info_text += f"<b>Reported Symptoms:</b> {', '.join(formatted_symptoms)}"
        self.patient_info.setText(patient_info_text)
        
        # Set diagnosis text
        self.diagnosis_text.setHtml(f"<h2>{predicted_disease}</h2>Based on your reported symptoms, our system suggests a possible diagnosis of {predicted_disease}.<br>Note: This is an automated assessment and not a definitive medical diagnosis.")
        
        # Set description text
        self.description_text.setHtml(f"<p>{desc}</p>")
        
        # Set precautions text
        precautions_html = "<ul>"
        for p_list in precautions:
            for p in p_list:
                # Handle potential NaN or float values from CSV
                p_str = str(p) if p is not None else ""
                if p_str and p_str.strip() and p_str.lower() != 'nan':
                    precautions_html += f"<li>{p_str.strip()}</li>"
        precautions_html += "</ul>"
        self.precautions_text.setHtml(precautions_html)
        
        # Set medications text
        medications_html = "<ul>"
        for m in medications:
            if m and m.strip():
                medications_html += f"<li>{m}</li>"
        medications_html += "</ul>"
        self.medications_text.setHtml(medications_html)
        
        # Set diet text
        diet_html = "<ul>"
        for d in diet:
            if d and d.strip():
                diet_html += f"<li>{d}</li>"
        diet_html += "</ul>"
        self.diet_text.setHtml(diet_html)
        
        # Set workout text
        workout_html = "<ul>"
        for w in workout:
            if w and w.strip():
                workout_html += f"<li>{w}</li>"
        workout_html += "</ul>"
        self.workout_text.setHtml(workout_html)
        
        # Switch to results page
        self.stacked_widget.setCurrentIndex(1)
    
    def go_back_to_form(self):
        """Return to the form page without clearing inputs"""
        self.stacked_widget.setCurrentIndex(0)
    
    def clear_and_reset(self):
        """Clear all form inputs and return to form page"""
        self.name_input.clear()
        self.age_input.clear()
        self.gender_input.setCurrentIndex(0)
        
        # Clear selected symptoms
        self.selected_symptoms_list.clear()
        
        # Uncheck all symptom checkboxes
        for checkbox in self.symptom_checks.values():
            checkbox.setChecked(False)
            
        # Return to form page
        self.stacked_widget.setCurrentIndex(0)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    medical_system = MedicalDiagnosisSystem()
    medical_system.show()
    sys.exit(app.exec())
