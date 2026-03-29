import tkinter as tk
import numpy as np
from sklearn.linear_model import LinearRegression

class MLGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple ML: Linear Regression Visualizer")
        
        self.points_x = []
        self.points_y = []
        
        # UI Setup
        self.label = tk.Label(root, text="Click on the white canvas to add data points!")
        self.label.pack(pady=10)
        
        self.canvas = tk.Canvas(root, width=600, height=400, bg="white", cursor="cross")
        self.canvas.pack(padx=20, pady=20)
        self.canvas.bind("<Button-1>", self.add_point)
        
        self.reset_btn = tk.Button(root, text="Clear Canvas", command=self.reset)
        self.reset_btn.pack(pady=10)

    def add_point(self, event):
        # Save the point coordinates
        self.points_x.append([event.x])
        self.points_y.append(event.y)
        
        # Draw the point (a small blue circle)
        self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="blue")
        
        # Run the ML Model if we have at least 2 points
        if len(self.points_x) > 1:
            self.train_and_draw()

    def train_and_draw(self):
        # 1. Prepare Data
        X = np.array(self.points_x)
        y = np.array(self.points_y)
        
        # 2. Create and Train the Model
        model = LinearRegression()
        model.fit(X, y)
        
        # 3. Predict line endpoints for the canvas width (0 to 600)
        x_range = np.array([[0], [600]])
        y_pred = model.predict(x_range)
        
        # 4. Update the GUI line
        self.canvas.delete("trendline")
        self.canvas.create_line(0, y_pred[0], 600, y_pred[1], fill="red", width=2, tags="trendline")

    def reset(self):
        self.points_x = []
        self.points_y = []
        self.canvas.delete("all")

if __name__ == "__main__":
    root = tk.Tk()
    app = MLGui(root)
    root.mainloop()