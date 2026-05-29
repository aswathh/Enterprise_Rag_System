# Base image
FROM python:3.11-slim

#set working directory
WORKDIR /app

# Copy requirements first 
COPY requirements.txt .

#Install dependencies
RUN pip install --upgrade pip setuptools wheel

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

#copy project files

COPY src/ ./src/
COPY Data/ ./Data/
COPY .env .
 
#Expose streamlit port
EXPOSE 8502

#Run the app
CMD ["streamlit", "run", "src/app.py", "--server.port=8502", "--server.address=0.0.0.0"]
