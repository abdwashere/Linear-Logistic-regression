import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ==================== DATA GENERATION (PAKISTANI CONTEXT) ====================
np.random.seed(42)
num_samples = 300

# Generating realistic distributions
matric_marks = np.random.randint(600, 1101, num_samples)
inter_marks = np.random.randint(550, 1101, num_samples)
test_score = np.random.randint(30, 101, num_samples)
interview_score = np.random.randint(1, 11, num_samples)

# Standard Pakistani Engineering Aggregate: 10% Matric, 40% Inter, 50% Test
matric_pct = (matric_marks / 1100) * 100
inter_pct = (inter_marks / 1100) * 100
test_pct = test_score 
aggregate = (matric_pct * 0.10) + (inter_pct * 0.40) + (test_pct * 0.50)

# Add noise for realism and set cutoff at ~65%
noise = np.random.normal(0, 2.0, num_samples)
admitted = ((aggregate + noise) >= 65).astype(int)

df = pd.DataFrame({
    'Matric_Marks': matric_marks,
    'Inter_Marks': inter_marks,
    'Test_Score': test_score,
    'Interview': interview_score,
    'Aggregate_%': np.round(aggregate, 2),
    'Admitted': admitted
})

# Train Logistic Regression
X = df[['Matric_Marks', 'Inter_Marks', 'Test_Score', 'Interview']]
y = df['Admitted']
model = LogisticRegression(max_iter=1000).fit(X, y)
accuracy = accuracy_score(y, model.predict(X))

# ==================== GUI FUNCTIONS ====================

def compute_agg(m_m, i_m, t_s):
    return ((m_m/1100)*10) + ((i_m/1100)*40) + (t_s * 0.5)

def predict_admission():
    try:
        m_m = float(entry_matric.get())
        i_m = float(entry_inter.get())
        t_s = float(entry_test.get())
        intv = float(entry_interview.get())

        if not (0 <= m_m <= 1100 and 0 <= i_m <= 1100 and 0 <= t_s <= 100):
            messagebox.showerror("Error", "Please enter marks within valid ranges.")
            return

        prob = model.predict_proba([[m_m, i_m, t_s, intv]])[0][1]
        agg_val = compute_agg(m_m, i_m, t_s)
        
        result = "ADMITTED" if prob >= 0.5 else "REJECTED"
        color = "#2ecc71" if result == "ADMITTED" else "#e74c3c"

        lbl_result.config(text=f"Decision: {result}", foreground=color)
        lbl_prob.config(text=f"Logistic Probability: {prob:.2%}  |  Aggregate: {agg_val:.2f}%")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

def show_logistic_insights():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Feature Importance (The Weights)
    importance = model.coef_[0]
    features = ['Matric', 'Inter', 'Entry Test', 'Interview']
    ax1.bar(features, importance, color=['#3498db', '#2ecc71', '#e67e22', '#9b59b6'], edgecolor='black')
    ax1.set_title("How the Model Weights Each Subject", fontweight='bold')
    ax1.set_ylabel("Impact on Probability (Coefficient)")
    
    # 2. The Sigmoid Curve (Probability Transformation)
    x_range = np.linspace(40, 95, 100).reshape(-1, 1)
    # Fit visual-only model for aggregate vs probability
    temp_model = LogisticRegression().fit(df[['Aggregate_%']], y)
    probs = temp_model.predict_proba(x_range)[:, 1]
    
    ax2.plot(x_range, probs, color='red', linewidth=3, label='Sigmoid Curve')
    ax2.scatter(df['Aggregate_%'], df['Admitted'], alpha=0.3, color='gray', label='Student Data')
    ax2.axhline(0.5, color='blue', linestyle='--', label='Decision Threshold (0.5)')
    ax2.set_xlabel("Calculated Aggregate %")
    ax2.set_ylabel("Probability of Admission")
    ax2.set_title("The Logistic Sigmoid Function", fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# ==================== MAIN GUI SETUP ====================
root = tk.Tk()
root.title("Pakistan Admission Predictor - Logistic Regression")
root.geometry("950x800")

style = ttk.Style()
style.theme_use('clam')

nb = ttk.Notebook(root)
nb.pack(fill='both', expand=True, padx=10, pady=10)

# --- TAB 1: PREDICTOR ---
tab1 = ttk.Frame(nb)
nb.add(tab1, text=" Admission Predictor ")

ttk.Label(tab1, text="University Admission Prediction System", font=("Arial", 18, "bold")).pack(pady=15)

input_box = ttk.LabelFrame(tab1, text=" Enter Student Data ", padding=20)
input_box.pack(padx=50, fill="x", pady=10)

# Input Fields
entries = []
fields = [("Matric Marks (out of 1100):", "950"), ("Inter Marks (out of 1100):", "920"), 
          ("Entry Test (0-100):", "75"), ("Interview (0-10):", "7")]

for i, (txt, val) in enumerate(fields):
    ttk.Label(input_box, text=txt).grid(row=i, column=0, sticky="w", pady=5)
    ent = ttk.Entry(input_box, width=15)
    ent.insert(0, val)
    ent.grid(row=i, column=1, padx=20, pady=5)
    entries.append(ent)

entry_matric, entry_inter, entry_test, entry_interview = entries

ttk.Button(tab1, text="Run Logistic Prediction", command=predict_admission).pack(pady=20)
lbl_result = ttk.Label(tab1, text="Decision: --", font=("Arial", 22, "bold"))
lbl_result.pack()
lbl_prob = ttk.Label(tab1, text="Logistic Probability: --", font=("Arial", 12))
lbl_prob.pack()

# --- TAB 2: DATASET ---
tab2 = ttk.Frame(nb)
nb.add(tab2, text=" Training Data ")

tree = ttk.Treeview(tab2, columns=list(df.columns), show='headings', height=18)
for col in df.columns:
    tree.heading(col, text=col)
    tree.column(col, width=120, anchor="center")
for _, row in df.iterrows():
    tree.insert("", "end", values=list(row))
tree.pack(fill="both", expand=True, padx=20, pady=20)

# --- TAB 3: MODEL INSIGHTS ---
tab3 = ttk.Frame(nb)
nb.add(tab3, text=" Logistic Regression Insights ")

ttk.Label(tab3, text="How the Math Works", font=("Arial", 16, "bold")).pack(pady=20)

math_frame = ttk.LabelFrame(tab3, text=" The Logistic Process ", padding=15)
math_frame.pack(padx=40, fill="x", pady=10)

math_info = (
    "1. Weighted Sum: Model assigns weights (coefficients) to each input.\n"
    "2. Log-Odds: A 'Z' score is calculated from your marks.\n"
    "3. Sigmoid Function: This score is squashed into a 0 to 1 probability.\n"
    "4. Decision: If probability is > 0.5, the student is admitted."
)
ttk.Label(math_frame, text=math_info, justify="left", font=("Arial", 11)).pack(anchor="w")

ttk.Button(tab3, text="Launch Mathematical Analysis", command=show_logistic_insights).pack(pady=25)

# Status Footer
status = ttk.Label(root, text=f" Model trained on {num_samples} records | Accuracy: {accuracy:.1%}", relief="sunken", anchor="w")
status.pack(side="bottom", fill="x")

root.mainloop()