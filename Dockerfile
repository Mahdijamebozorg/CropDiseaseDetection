# Dockerfile

# Set the base image
FROM tensorflow/tensorflow:2.10.1

# Install the required dependencies (gcc)
# RUN apk add build-base

# RUN apt-get install build-essential

# Set the working directory
WORKDIR /app

# Confingure Python using environmental variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy the requirements file into the image and install them
COPY ./requirements.txt .
# RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir --upgrade -r ./requirements.txt

# Copy the source code into the image
COPY . .

# Expose the port
EXPOSE 8000

# Start the Uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]