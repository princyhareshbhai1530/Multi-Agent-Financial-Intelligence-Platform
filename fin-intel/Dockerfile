FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: serve the dashboard. Override CMD to run `python run.py` for batch mode.
EXPOSE 8501
CMD ["streamlit", "run", "app/dashboard.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
