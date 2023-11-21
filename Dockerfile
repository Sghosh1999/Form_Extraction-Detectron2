# Use a specific Python version
FROM mcr.microsoft.com/azure-functions/python:4-python3.8

RUN adduser --system --no-create-home k1_nonroot
ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true

# Install system dependencies
RUN apt-get update -y && apt-get install -y \
    poppler-utils \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx \
    libtesseract-dev tesseract-ocr
RUN apt-get install libmagickwand-dev -y

RUN apt-get install aptitude -y
RUN aptitude install -y ghostscript
RUN apt-get install -y tesseract-ocr


RUN pip install --upgrade pip
RUN apt-get install -y libtasn1-6
RUN apt-get install -y curl
RUN apt-get update
RUN apt-get install -y openssl
RUN apt-get install -y build-essential libapr1-dev libssl-dev
RUN apt-get update && apt-get install apt-utils wget build-essential cmake libfreetype6-dev pkg-config libfontconfig-dev libjpeg-dev libopenjp2-7-dev libcairo2-dev libtiff5-dev -y
RUN apt-get install ca-certificates
RUN apt-get -y install cmake libblkid-dev e2fslibs-dev libboost-all-dev libaudit-dev


##Setup Dectron2



# Set the working directory
WORKDIR /home/site/wwwroot/data_harvester




# Copy and install Python dependencies
COPY requirements.txt /home/site/wwwroot/data_harvester
RUN pip install -r requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential


RUN rm -r -f /var/lib/apt/lists/*

# RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install cython opencv-python

RUN git clone https://github.com/facebookresearch/detectron2.git

RUN CFLAGS="-Wno-narrowing" pip install -e detectron2  # Skip CUDA-dependent components



# Copy the application code
COPY . /home/site/wwwroot/data_harvester
# RUN chmod -R 777 /home/site/wwwroot
RUN chown -R k1_nonroot /home/site/wwwroot/data_harvester
RUN mkdir -p /home/.EasyOCR
RUN chown -R k1_nonroot /home/.EasyOCR
RUN chmod -R 777 /home/.EasyOCR
# RUN chown -R k1_nonroot /home
USER k1_nonroot
# Expose the port Flask will be running on
EXPOSE 8000

# Run the command to start the Flask app
CMD ["python", "app.py"]
