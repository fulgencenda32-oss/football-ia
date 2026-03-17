FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/

RUN mkdir -p /app/historique && echo '[]' > /app/historique/predictions.json

EXPOSE 7860

ENV PORT=7860

CMD ["python", "/app/app.py"]