# Use the official Python image as a base image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies including libGL
RUN apt-get update && apt-get install -y espeak

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgl1-mesa-glx libgl1-mesa-dri libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker caching
COPY requirements.txt /app/

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . /app

# Expose port 5000 for the Flask app
EXPOSE 5000

# Set environment variables for OpenCV to not require a display (headless)
ENV DISPLAY=:0

# Run the Flask app
CMD ["python", "app.py"]
