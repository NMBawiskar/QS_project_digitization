FROM python:3.8
RUN apt-get update && apt-get install wget build-essential cmake libfreetype6-dev pkg-config libfontconfig-dev libjpeg-dev libgs-dev libopenjp2-7-dev libgl1-mesa-glx tesseract-ocr libtesseract-dev -y
RUN apt-get install poppler-utils libpoppler-cpp-dev -y
WORKDIR /digitization
RUN pip install --upgrade pip setuptools wheel
EXPOSE 5001
ADD requirements.txt /digitization/requirements.txt
RUN pip install -r requirements.txt
RUN python -m nltk.downloader words
COPY . .
CMD ["python", "server-manager.py"]