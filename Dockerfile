# Base image
FROM python:3.11-slim

#set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy requirements first 
COPY requirements.txt .

#Install dependencies
RUN uv pip install --system -r requirements.txt

#copy project files

COPY src/ ./src/
COPY Data/ ./Data/
COPY .env .
 
#Expose streamlit port
EXPOSE 8501

#Run the app
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
