FROM python:3.10-slim

# Install necessary packages
RUN apt-get update && apt-get install -y \
x11-apps \
x11-xserver-utils \
libgl1-mesa-glx \
libglib2.0-0 \
libsm6 \
libxext6 \
libxrender1 \
curl \ 
&& rm -rf /var/lib/apt/lists/*

# Install DIAMBRA Arena
RUN pip install diambra-arena

# Copy your script into the container
COPY 3rdstrikeailetsgo.py /app/3rdstrikeailetsgo.py

# Set the working directory
WORKDIR /app

# Set the display environment variable
ENV DISPLAY=:0

# Command to run your script
CMD ["python", "3rdstrikeailetsgo.py"]