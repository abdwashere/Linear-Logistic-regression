import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import json
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# ==================== SETUP DATA AND MODEL ====================
data = {
    'Hours_Studied': [2, 9, 5, 1, 8, 10, 4, 7, 3, 6, 12, 2, 5, 9, 1],
    'Prev_Grade':    [60, 85, 70, 50, 80, 95, 65, 78, 62, 75, 98, 55, 68, 88, 52],
    'Sleep_Hours':   [8, 7, 6, 5, 8, 7, 9, 6, 7, 8, 6, 5, 7, 8, 4],
    'Final_Mark':    [62, 92, 72, 45, 88, 98, 68, 81, 65, 79, 100, 58, 71, 95, 48]
}
df = pd.DataFrame(data)
X = df[['Hours_Studied', 'Prev_Grade', 'Sleep_Hours']]
y = df['Final_Mark']
model = LinearRegression().fit(X, y)

# Calculate model metrics
r2 = r2_score(y, model.predict(X))
mae = mean_absolute_error(y, model.predict(X))

# ==================== GUI FUNCTIONS ====================
def predict_mark():
    try:
        h = float(entry_hours.get())
        p = float(entry_prev.get())
        s = float(entry_sleep.get())
        
        # Input validation
        if not (0 <= h <= 24):
            messagebox.showerror("Input Error", "Hours studied should be between 0-24")
            return
        if not (0 <= p <= 100):
            messagebox.showerror("Input Error", "Previous grade should be between 0-100")
            return
        if not (0 <= s <= 24):
            messagebox.showerror("Input Error", "Sleep hours should be between 0-24")
            return
            
        prediction = model.predict([[h, p, s]])[0]
        prediction = max(0, min(100, prediction))  # Clamp between 0-100
        lbl_result.config(text=f"Predicted Mark: {prediction:.2f}/100", foreground="green")
        
        # Show feature importance
        update_feature_importance()
        
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers.")

def save_prediction():
    if lbl_result.cget("text") == "Predicted Mark: --":
        messagebox.showwarning("No Prediction", "Please make a prediction first!")
        return
        
    prediction_data = {
        'hours': entry_hours.get(),
        'prev_grade': entry_prev.get(),
        'sleep': entry_sleep.get(),
        'prediction': lbl_result.cget('text'),
        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        with open('predictions.json', 'r') as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    
    history.append(prediction_data)
    
    with open('predictions.json', 'w') as f:
        json.dump(history[-20:], f, indent=2)  # Keep last 20 predictions
    
    messagebox.showinfo("Saved", "Prediction saved to history!")
    update_history_display()

def clear_inputs():
    entry_hours.delete(0, tk.END)
    entry_prev.delete(0, tk.END)
    entry_sleep.delete(0, tk.END)
    lbl_result.config(text="Predicted Mark: --", foreground="black")

def show_enhanced_graph():
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Student Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Hours Studied vs Final Mark
    axes[0, 0].scatter(df['Hours_Studied'], df['Final_Mark'], color='blue', alpha=0.6)
    axes[0, 0].set_xlabel("Hours Studied", fontsize=10)
    axes[0, 0].set_ylabel("Final Mark", fontsize=10)
    axes[0, 0].set_title("Study Hours vs Marks", fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['Hours_Studied'], df['Final_Mark'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df['Hours_Studied'].sort_values(), 
                    p(df['Hours_Studied'].sort_values()), 
                    "r-", alpha=0.8, label=f'Trend (slope: {z[0]:.2f})')
    axes[0, 0].legend()
    
    # 2. Previous Grade vs Final Mark
    axes[0, 1].scatter(df['Prev_Grade'], df['Final_Mark'], color='green', alpha=0.6)
    axes[0, 1].set_xlabel("Previous Grade", fontsize=10)
    axes[0, 1].set_ylabel("Final Mark", fontsize=10)
    axes[0, 1].set_title("Previous Grade vs Marks", fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Sleep Hours vs Final Mark
    axes[1, 0].scatter(df['Sleep_Hours'], df['Final_Mark'], color='orange', alpha=0.6)
    axes[1, 0].set_xlabel("Sleep Hours", fontsize=10)
    axes[1, 0].set_ylabel("Final Mark", fontsize=10)
    axes[1, 0].set_title("Sleep Hours vs Marks", fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Feature Correlation Heatmap (simplified)
    corr_matrix = df.corr()
    im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(corr_matrix.columns)))
    axes[1, 1].set_yticks(range(len(corr_matrix.columns)))
    axes[1, 1].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    axes[1, 1].set_yticklabels(corr_matrix.columns)
    axes[1, 1].set_title("Feature Correlations", fontsize=12)
    
    # Add correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=axes[1, 1])
    plt.tight_layout()
    plt.show()

def update_feature_importance():
    # Display feature coefficients
    features = ['Hours Studied', 'Previous Grade', 'Sleep Hours']
    coefs = model.coef_
    
    importance_text = "Feature Impact:\n"
    for feat, coef in zip(features, coefs):
        impact = "Positive" if coef > 0 else "Negative"
        importance_text += f"• {feat}: {coef:.3f} ({impact})\n"
    
    lbl_importance.config(text=importance_text)

def update_history_display():
    try:
        with open('predictions.json', 'r') as f:
            history = json.load(f)
        
        # Clear existing items
        for item in history_tree.get_children():
            history_tree.delete(item)
        
        # Add last 5 predictions
        for pred in history[-5:]:
            values = (
                pred.get('hours', ''),
                pred.get('prev_grade', ''),
                pred.get('sleep', ''),
                pred.get('prediction', '').replace('Predicted Mark: ', ''),
                pred.get('timestamp', '')
            )
            history_tree.insert("", "end", values=values)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

def show_model_metrics():
    metrics_text = f"Model Performance Metrics:\n"
    metrics_text += f"• R² Score: {r2:.3f} (1.0 = perfect prediction)\n"
    metrics_text += f"• Mean Absolute Error: {mae:.2f} points\n"
    metrics_text += f"• Model Intercept: {model.intercept_:.2f}\n"
    messagebox.showinfo("Model Metrics", metrics_text)

# ==================== CREATE MAIN WINDOW ====================
root = tk.Tk()
root.title("Student Performance Predictor - Enhanced")
root.geometry("900x800")
root.resizable(True, True)

# Configure style
style = ttk.Style()
style.theme_use('clam')
style.configure('TButton', font=('Arial', 10))
style.configure('TLabel', font=('Arial', 10))
style.configure('Header.TLabel', font=('Arial', 14, 'bold'))

# ==================== CREATE NOTEBOOK FOR TABS ====================
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True, padx=10, pady=5)

# ==================== TAB 1: PREDICTION ====================
prediction_frame = ttk.Frame(notebook)
notebook.add(prediction_frame, text='Prediction')

# Title
title_label = ttk.Label(prediction_frame, text="Student Performance Predictor", 
                       style='Header.TLabel')
title_label.pack(pady=10)

# Input Frame
input_frame = ttk.LabelFrame(prediction_frame, text="Input Parameters", padding=15)
input_frame.pack(pady=10, padx=20, fill="x")

# Hours Studied
ttk.Label(input_frame, text="Hours Studied (0-24):").grid(row=0, column=0, sticky='w', pady=5)
entry_hours = ttk.Entry(input_frame, width=20)
entry_hours.grid(row=0, column=1, pady=5, padx=10)
entry_hours.insert(0, "5")

# Previous Grade
ttk.Label(input_frame, text="Previous Grade (0-100):").grid(row=1, column=0, sticky='w', pady=5)
entry_prev = ttk.Entry(input_frame, width=20)
entry_prev.grid(row=1, column=1, pady=5, padx=10)
entry_prev.insert(0, "70")

# Sleep Hours
ttk.Label(input_frame, text="Sleep Hours (0-24):").grid(row=2, column=0, sticky='w', pady=5)
entry_sleep = ttk.Entry(input_frame, width=20)
entry_sleep.grid(row=2, column=1, pady=5, padx=10)
entry_sleep.insert(0, "7")

# Button Frame
button_frame = ttk.Frame(prediction_frame)
button_frame.pack(pady=10)

ttk.Button(button_frame, text="Predict Mark", command=predict_mark).pack(side='left', padx=5)
ttk.Button(button_frame, text="Clear Inputs", command=clear_inputs).pack(side='left', padx=5)
ttk.Button(button_frame, text="Save Prediction", command=save_prediction).pack(side='left', padx=5)
ttk.Button(button_frame, text="Model Metrics", command=show_model_metrics).pack(side='left', padx=5)

# Result Label
lbl_result = ttk.Label(prediction_frame, text="Predicted Mark: --", 
                       font=("Arial", 14, "bold"), foreground="blue")
lbl_result.pack(pady=10)

# Feature Importance
lbl_importance = ttk.Label(prediction_frame, text="Feature Impact:\n• Hours Studied: --\n• Previous Grade: --\n• Sleep Hours: --",
                          justify='left', font=("Arial", 10))
lbl_importance.pack(pady=10)

# ==================== TAB 2: TRAINING DATA ====================
data_frame = ttk.Frame(notebook)
notebook.add(data_frame, text='Training Data')

# Training Data Table
ttk.Label(data_frame, text="Complete Training Dataset", 
          font=("Arial", 12, "bold")).pack(pady=10)

# Create Treeview with scrollbar
tree_frame = ttk.Frame(data_frame)
tree_frame.pack(fill='both', expand=True, padx=20, pady=10)

tree_scroll = ttk.Scrollbar(tree_frame)
tree_scroll.pack(side='right', fill='y')

tree = ttk.Treeview(tree_frame, columns=list(df.columns), show='headings', 
                    height=12, yscrollcommand=tree_scroll.set)
tree_scroll.config(command=tree.yview)

for col in df.columns:
    tree.heading(col, text=col)
    tree.column(col, width=120, anchor='center')

for index, row in df.iterrows():
    tree.insert("", "end", values=list(row))

tree.pack(side='left', fill='both', expand=True)

# Data Statistics
stats_frame = ttk.LabelFrame(data_frame, text="Data Statistics", padding=10)
stats_frame.pack(fill='x', padx=20, pady=10)

stats_text = f"Total Records: {len(df)}\n"
stats_text += f"Average Study Hours: {df['Hours_Studied'].mean():.2f}\n"
stats_text += f"Average Previous Grade: {df['Prev_Grade'].mean():.2f}\n"
stats_text += f"Average Sleep Hours: {df['Sleep_Hours'].mean():.2f}\n"
stats_text += f"Average Final Mark: {df['Final_Mark'].mean():.2f}"

ttk.Label(stats_frame, text=stats_text, justify='left').pack()

# ==================== TAB 3: VISUALIZATION ====================
viz_frame = ttk.Frame(notebook)
notebook.add(viz_frame, text='Visualization')

viz_label = ttk.Label(viz_frame, text="Performance Visualizations", 
                      font=("Arial", 12, "bold"))
viz_label.pack(pady=10)

# Graph buttons
graph_frame = ttk.Frame(viz_frame)
graph_frame.pack(pady=20)

ttk.Button(graph_frame, text="Show Enhanced Analysis", 
           command=show_enhanced_graph, width=25).pack(pady=5)

# Simple correlation display
corr_label = ttk.Label(viz_frame, text="Feature Correlations with Final Mark:", 
                       font=("Arial", 10, "bold"))
corr_label.pack(pady=10)

correlations = df.corr()['Final_Mark'].drop('Final_Mark')
corr_text = ""
for feature, corr in correlations.items():
    strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
    corr_text += f"• {feature}: {corr:.3f} ({strength} correlation)\n"

ttk.Label(viz_frame, text=corr_text, justify='left').pack()

# ==================== TAB 4: HISTORY ====================
history_frame = ttk.Frame(notebook)
notebook.add(history_frame, text='History')

ttk.Label(history_frame, text="Recent Predictions History", 
          font=("Arial", 12, "bold")).pack(pady=10)

# History Treeview
hist_tree_frame = ttk.Frame(history_frame)
hist_tree_frame.pack(fill='both', expand=True, padx=20, pady=10)

hist_scroll = ttk.Scrollbar(hist_tree_frame)
hist_scroll.pack(side='right', fill='y')

history_tree = ttk.Treeview(hist_tree_frame, 
                            columns=('Hours', 'Prev Grade', 'Sleep', 'Prediction', 'Time'),
                            show='headings', height=8, yscrollcommand=hist_scroll.set)
hist_scroll.config(command=history_tree.yview)

history_tree.heading('Hours', text='Hours')
history_tree.heading('Prev Grade', text='Prev Grade')
history_tree.heading('Sleep', text='Sleep')
history_tree.heading('Prediction', text='Prediction')
history_tree.heading('Time', text='Timestamp')

history_tree.column('Hours', width=80)
history_tree.column('Prev Grade', width=80)
history_tree.column('Sleep', width=80)
history_tree.column('Prediction', width=100)
history_tree.column('Time', width=150)

history_tree.pack(side='left', fill='both', expand=True)

# Refresh button
ttk.Button(history_frame, text="Refresh History", 
           command=update_history_display).pack(pady=10)

# Instructions
instructions = ttk.Label(history_frame, 
                        text="Make predictions and click 'Save Prediction' to add to history",
                        font=("Arial", 9, "italic"))
instructions.pack(pady=5)

# ==================== STATUS BAR ====================
status_frame = ttk.Frame(root)
status_frame.pack(fill='x', side='bottom', padx=10, pady=5)

status_label = ttk.Label(status_frame, 
                        text=f"Model ready | R²: {r2:.3f} | MAE: {mae:.2f} | Data points: {len(df)}")
status_label.pack(side='left')

# Initialize history display
update_history_display()
update_feature_importance()

# ==================== START APPLICATION ====================
root.mainloop()