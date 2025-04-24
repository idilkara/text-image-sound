FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y ffmpeg

# Copy the rest 
COPY . .

# Expose the port app runs on
EXPOSE 8080

CMD ["python3", "app.py"]