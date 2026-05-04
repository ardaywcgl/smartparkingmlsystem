AI-powered smart parking prediction system using machine learning and real-time adjustments.

# Smart Parking Prediction System

This project is a **hybrid** system that predicts parking lot occupancy rates by combining machine learning and real-time data correction algorithms.

---

## 🚀 Key Features

- **🧠 Hybrid Prediction Engine**: 
  - **Random Forest**: Baseline prediction based on historical trends.
  - **Gaussian Event Impact**: Models the time-based impact of special events (matches, concerts, etc.) on parking density.
  - **Exponential Decay**: Instantly corrects the difference between real-time occupancy data and predictions.
- **📊 Modern Dashboard**: Sleek and responsive interface developed for both users and administrators.
- **⚙️ Dynamic Management**: Ability to add/remove events and manually simulate real-time parking status.

---

## 📦 Installation and Execution

Follow these steps to run the project on your local machine:

1. **Clone the Project**:
   ```bash
   git clone https://github.com/username/smart-parking-ml.git
   cd smart-parking-ml
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model (Important)**:
   The system does not come with a default model. The model must be trained for full performance:
   ```bash
   python train_model.py
   ```
   *If the model is missing, the system will automatically continue to run in "Fallback" mode (with default values).*

4. **Start the Application**:
   ```bash
   python app.py
   ```
   You can start using the system by navigating to `http://127.0.0.1:5000` in your browser.

---

## ⚙️ How the System Works

The prediction engine operates with a three-stage logic:
1. **Base ML Prediction**: The model analyzes hour, day, and past occupancy trends.
2. **Event Impact**: The density of defined events is calculated using Gaussian distribution (bell curve).
3. **Real-time Correction**: The difference between the actual current occupancy of the parking lot and the prediction is projected into the future by damping over time (exponential decay).

---

## 📂 Project Structure

- `app.py`: Main Flask server and prediction engine logic.
- `train_model.py`: Model training and backtest script.
- `templates/`: HTML interface files.
- `static/`: CSS and styling files.
- `tools/`: 🧪 Contains helper scripts for data analysis and visualization (e.g., hourly occupancy analysis).
- `data.json`: Stores the current status of the system and events. Created automatically on first run.

---

## 📊 Dataset

The `dataset.csv` used in the project is a public dataset containing parking data for the city of Birmingham.

---
*This project was developed within the scope of smart city technologies and data science applications.*
