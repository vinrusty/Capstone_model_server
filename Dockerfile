# Use the official Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the ML_models directory to the working directory
COPY ML_models /app/ML_models

# Copy the RL_models directory to the working directory
COPY RL_models /app/RL_models

# Copy the main.py file to the working directory
COPY main.py /app

# Set the PYTHONPATH environment variable to include RL_models directory
ENV PYTHONPATH "${PYTHONPATH}:/app/ML_models:/app/RL_models"

# Copy the requirements file to the working directory
COPY requirements.txt /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
