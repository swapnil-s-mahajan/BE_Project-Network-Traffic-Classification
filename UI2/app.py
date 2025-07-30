from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import torch
import joblib
import os
import matplotlib.pyplot as plt
from model import CNNClassifier  # import model class
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 69  # Update if your model input dimension is different
num_classes = 17  # Update based on your actual class count

# Load model and scaler
model = CNNClassifier(input_dim, num_classes).to(device)
model.load_state_dict(torch.load("fgan2_classifier_dec.pth", map_location=device))
model.eval()

scaler = joblib.load("scaler_fgani.pkl")

label_map = {
    0: 'AMAZON', 1: 'CLOUDFLARE', 2: 'DROPBOX', 3: 'FACEBOOK',
    4: 'GMAIL', 5: 'GOOGLE', 6: 'HTTP', 7: 'HTTP_CONNECT', 8: 'HTTP_PROXY',
    9: 'MICROSOFT', 10: 'MSN', 11: 'SKYPE', 12: 'SSL', 13: 'TWITTER',
    14: 'WINDOWS_UPDATE', 15: 'YAHOO', 16: 'YOUTUBE'
}


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename.endswith(".csv"):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            return redirect(url_for("result", filename=file.filename))
    return render_template("upload.html")

# @app.route("/result/<filename>")
# def result(filename):
#     filepath = os.path.join(UPLOAD_FOLDER, filename)
#     df = pd.read_csv(filepath)

#     X_scaled = scaler.transform(df.values)
#     X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

#     with torch.no_grad():
#         preds = model(X_tensor)
#         predicted_classes = preds.argmax(dim=1).cpu().numpy()
    
#     # Assuming predictions is a NumPy array of class indices
#     predicted_labels = [label_map[int(p)] for p in predicted_classes]

#     df["Predicted_Class"] = predicted_labels

#     # Plot pie chart
#     class_counts = df["Predicted_Class"].value_counts()
#     plt.figure(figsize=(6, 6))
#     plt.pie(class_counts, labels=class_counts.index, autopct="%1.1f%%", startangle=140)
#     plt.title("Traffic Classification")
#     plot_path = os.path.join("static", "plot.png")
#     plt.savefig(plot_path)
#     plt.close()

#     # Render result page
#     return render_template("result.html", tables=[df.head(10).to_html(classes="data")], plot_url=plot_path)

@app.route("/result")
def result_no_file():
    return render_template("result_empty.html")


@app.route("/result/<filename>")
def result(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    df = pd.read_csv(filepath)

    X_scaled = scaler.transform(df.values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X_tensor)
        predicted_classes = preds.argmax(dim=1).cpu().numpy()
    
    # Assuming predictions is a NumPy array of class indices
    predicted_labels = [label_map[int(p)] for p in predicted_classes]

    df["Predicted_Class"] = predicted_labels

    # Class count for pie and bar charts
    class_counts = df["Predicted_Class"].value_counts().to_dict()

    # Plot pie chart
    class_labels = list(class_counts.keys())
    class_data = list(class_counts.values())

    # Save the class counts for passing to the frontend
    class_counts_json = class_counts

    # Plot pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(class_data, labels=class_labels, autopct="%1.1f%%", startangle=140, 
            colors=['#8ECAE6', '#219EBC', '#023047', '#FFB703', '#FB8500'])
    plt.title("Traffic Classification")
    plot_path = os.path.join("static", "plot.png")
    plt.savefig(plot_path)
    plt.close()

    # Render result page with class counts
    return render_template("result.html", tables=[df.head(10).to_html(classes="data")], 
                           plot_url=plot_path, class_counts=class_counts_json)


if __name__ == "__main__":
    app.run(debug=True)
