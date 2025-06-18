FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /

# Install Dependencies
COPY requirements.txt .
COPY test_input.json .
RUN pip install --no-cache-dir -r requirements.txt

# Set serialized model paths
ENV DLV3P_MODEL_PATH='/models/dlv3p_model.pth'
ENV EFFDET_MODEL_PATH='/models/effdet_model.pth'

# Copy your handler file
COPY runpod_handler.py /
COPY src /src
COPY models /models

# Start the container
CMD ["python3", "-u", "runpod_handler.py"]