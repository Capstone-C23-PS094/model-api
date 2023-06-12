FROM python:3.9-slim-bullseye

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip

# Install dependencies:
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY  . .

# Run the application:
# CMD ["python", "-u", "main.py"]
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "main:app"]