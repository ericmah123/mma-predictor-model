import tkinter as tk
from tkinter import messagebox, ttk
import joblib
import pandas as pd
import os

class FightPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MMA Fight Predictor")

        # Load the trained model and scaler
        model_path = os.path.join(os.path.dirname(__file__), 'mma_fight_predictor.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
        self.model = joblib.load(model_path)
        self.scaler, self.features = joblib.load(scaler_path)

        # Create input fields for Fighter 1
        self.create_input_fields("Fighter 1", 0)
        
        # Create input fields for Fighter 2
        self.create_input_fields("Fighter 2", 10)

        # Create predict button
        predict_button = tk.Button(root, text="Predict", command=self.predict)
        predict_button.grid(row=20, column=0, columnspan=2, pady=10)

    def create_input_fields(self, label, start_row):
        tk.Label(self.root, text=label).grid(row=start_row, column=0, columnspan=2, pady=5)

        attributes = [
            "Height (in inches):", "Weight (in pounds):", "Reach (in inches):", 
            "Significant Strikes Landed Per Minute:", "Average Takedowns Landed Per 15 Minutes:",
            "Win/Loss Ratio:", "Experience:", "Age:"
        ]
        
        self.entries = getattr(self, 'entries', [])
        current_entries = []
        
        for i, attr in enumerate(attributes):
            tk.Label(self.root, text=attr).grid(row=start_row + i + 1, column=0, sticky='e', padx=5, pady=2)
            entry = tk.Entry(self.root)
            entry.grid(row=start_row + i + 1, column=1, padx=5, pady=2)
            current_entries.append(entry)
        
        # Add stance dropdown
        tk.Label(self.root, text="Stance:").grid(row=start_row + len(attributes) + 1, column=0, sticky='e', padx=5, pady=2)
        stance_var = tk.StringVar()
        stance_dropdown = ttk.Combobox(self.root, textvariable=stance_var)
        stance_dropdown['values'] = ('Orthodox', 'Southpaw')
        stance_dropdown.grid(row=start_row + len(attributes) + 1, column=1, padx=5, pady=2)
        current_entries.append(stance_dropdown)

        self.entries.append(current_entries)

    def predict(self):
        try:
            # Retrieve the inputs
            fighter_1_stats = [float(entry.get()) for entry in self.entries[0][:-1]]
            fighter_2_stats = [float(entry.get()) for entry in self.entries[1][:-1]]

            # Encode stance
            fighter_1_stance = self.entries[0][-1].get()
            fighter_2_stance = self.entries[1][-1].get()
            
            fighter_1_stats.append(1 if fighter_1_stance == 'Southpaw' else 0)
            fighter_2_stats.append(1 if fighter_2_stance == 'Southpaw' else 0)

            # Fill missing columns with 0 for consistency
            fighter_1_stats.append(0 if fighter_1_stance == 'Orthodox' else 0)
            fighter_2_stats.append(0 if fighter_2_stance == 'Orthodox' else 0)

            # Create the input DataFrame for the prediction
            fighter_1_df = pd.DataFrame([fighter_1_stats], columns=self.features)
            fighter_2_df = pd.DataFrame([fighter_2_stats], columns=self.features)

            # Scale the input data
            fighter_1_df = self.scaler.transform(fighter_1_df)
            fighter_2_df = self.scaler.transform(fighter_2_df)

            # Predict the outcome for both fighters
            fighter_1_prediction = self.model.predict_proba(fighter_1_df)[0][1]
            fighter_2_prediction = self.model.predict_proba(fighter_2_df)[0][1]

            # Calculate the prediction odds
            fighter_1_odds = fighter_1_prediction / (fighter_1_prediction + fighter_2_prediction)
            fighter_2_odds = fighter_2_prediction / (fighter_1_prediction + fighter_2_prediction)

            # Display the result
            result = f"Fighter 1 Win Probability: {fighter_1_odds * 100:.2f}%\nFighter 2 Win Probability: {fighter_2_odds * 100:.2f}%"
            messagebox.showinfo("Prediction Result", result)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values for all fields.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FightPredictorApp(root)
    root.mainloop()
