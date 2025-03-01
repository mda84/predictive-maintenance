# Predictive Maintenance for Industrial Equipment

## Overview
This project is designed to predict equipment failures before they occur by analyzing time-series sensor data (e.g., temperature, vibration, pressure) collected from industrial equipment. The system comprises several key components:

- **Data Pipeline:** Ingest, preprocess, and transform raw sensor data for analysis and model training.
- **Model Development:** Design, train, and optimize machine learning models (e.g., using LSTM or Transformer architectures) to predict equipment failures.
- **Model Deployment:** Serve the predictive model via a REST API built with FastAPI, containerized with Docker, and deployable on Kubernetes for scalable, production-level inference.
- **Performance Monitoring:** Log predictions and system performance, with continuous monitoring and alerting (via Prometheus/Grafana or similar tools).
- **Documentation & Collaboration:** Clear code organization, CI/CD pipelines, and interactive notebooks for research and experimentation.

This project demonstrates expertise in ML engineering, data pipelines, model deployment, and performance monitoring, and is intended as a portfolio piece for production-level systems.

## Project Structure
```
predictive-maintenance/ 
├── README.md # Overview, installation, usage, and deployment instructions. 
├── requirements.txt # Python package dependencies. ├── Dockerfile # Container configuration for model deployment. 
├── kubernetes/
│ ├── deployment.yaml # Kubernetes Deployment manifest for the FastAPI app. 
│ ├── service.yaml # Kubernetes Service manifest to expose the FastAPI app. 
│ └── postgres.yaml # (Optional) Kubernetes manifest for a PostgreSQL database for logging. 
├── data_pipeline/
│ ├── ingest.py # Script to ingest and clean raw sensor data. 
│ ├── transform.py # Data transformation and feature engineering. 
│ └── config.yaml # Configuration for the data pipeline. 
├── model/
│ ├── train.py # Script for training and optimizing the predictive model. 
│ ├── evaluate.py # Model evaluation and performance reporting. 
│ ├── predict.py # Inference script to generate predictions from sensor data. 
│ └── lstm_model.pth # (Output) Serialized model after training. 
├── api/
│ ├── app.py # FastAPI app serving predictions and logging performance. 
│ └── database.py # SQLAlchemy models and database connection setup. 
└── notebooks/
    └── Maintenance_Model_Development.ipynb # Notebook for interactive model experimentation and evaluation.
```

## Installation

### Prerequisites
- Python 3.9 or later.
- A virtual environment is recommended to isolate dependencies.

### Setup

**Clone the Repository:**
```
   git clone https://github.com/mda84/predictive-maintenance.git
   cd predictive-maintenance
```

Create and Activate a Virtual Environment:
```
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Install Dependencies:
```
pip install -r requirements.txt
```

## Usage
Data Pipeline

Ingest and Preprocess Data:
Use the data_pipeline/ingest.py script to load raw sensor data from a CSV file.

Run:
```
python data_pipeline/ingest.py path/to/sensor_data.csv
```

Transform Data:
Use the data_pipeline/transform.py script to normalize and feature-engineer the data.

Run:
```
python data_pipeline/transform.py path/to/sensor_data.csv feature1 feature2 ...
```

## Model Development
Training:
Run the training script to build and optimize the predictive model.
```
python model/train.py
```

The trained model will be saved to model/lstm_model.pth.

Evaluation:
Evaluate model performance with:
```
python model/evaluate.py model/lstm_model.pth path/to/sensor_data.csv
```

Prediction:
Use the prediction script to generate predictions from new sensor readings:
```
python model/predict.py feature1 feature2 ... 
```

## API Deployment
Running Locally:
Start the FastAPI server:
```
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

The prediction endpoint is available at http://localhost:8000/predict.

## Interactive Notebook
Open the Jupyter Notebook in the notebooks/ folder for interactive development:
```
jupyter notebook notebooks/Maintenance_Model_Development.ipynb
```

## Docker Deployment
Build the Docker Image:
```
docker build -t predictive-maintenance .
```

Run the Docker Container:
```
docker run -it -p 8000:8000 predictive-maintenance
```

## Kubernetes Deployment
The kubernetes/ directory contains sample manifests for deploying the FastAPI app and a PostgreSQL database:
```
kubectl apply -f kubernetes/postgres.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

## Performance Monitoring & Logging
The FastAPI app logs predictions and response times to a database (configured via DATABASE_URL in the environment). For production, integrate Prometheus and Grafana for real-time monitoring and alerts.

## Collaboration & CI/CD
Version Control: Use Git for tracking changes.

CI/CD: Set up GitHub Actions (or your preferred CI/CD tool) to run tests and build Docker images on code commits.

Documentation: This README and inline code comments document the project structure and usage, facilitating collaboration among data scientists, engineers, and product teams.

## Research and Experimentation
The interactive notebook provides a platform for experimenting with different model architectures, preprocessing techniques, and evaluation metrics. Continuous research and iteration are encouraged to enhance model performance.

## License
This project is licensed under the MIT License.

## Contact
For questions, collaboration, or contributions, please contact dorkhah9@gmail.com
