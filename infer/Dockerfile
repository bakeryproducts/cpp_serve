FROM python:3.8

COPY infer/whl /infer/whl
RUN pip3 install torchvision torch --no-index --find-links /infer/whl
# RUN pip3 install torchvision --no-index --find-links /infer/whl

ADD infer/requirements.txt /infer/requirements.txt
WORKDIR /infer
RUN pip3 install --no-cache-dir -r requirements.txt

ADD infer /infer
