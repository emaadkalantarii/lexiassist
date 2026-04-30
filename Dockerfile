# Use official Python 3.11 slim image as the base.
# 'slim' is a minimal version — smaller image size, faster builds.
FROM python:3.11-slim

# Set the working directory inside the container.
# All subsequent commands run from this path.
WORKDIR /app

# Set environment variables that improve Python behavior in containers.
# PYTHONDONTWRITEBYTECODE: stops Python from writing .pyc bytecode files
# PYTHONUNBUFFERED: forces stdout/stderr to flush immediately — important
# for seeing logs in real time from a running container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy only requirements first — before copying the rest of the code.
# Docker caches each layer. If requirements.txt hasn't changed,
# Docker reuses the cached pip install layer on rebuilds, saving time.
COPY requirements.txt .

# Install Python dependencies.
# --no-cache-dir: don't store pip's download cache — keeps image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container.
COPY . .

# Expose port 8000 so Docker knows the container listens on this port.
# This doesn't actually publish the port — that's done at runtime.
EXPOSE 8000

# The default command to run when the container starts.
# Runs the FastAPI app with uvicorn on all network interfaces (0.0.0.0)
# so it's accessible from outside the container.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]