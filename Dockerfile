# base image
FROM python:3.9-slim

# working directory
WORKDIR /app

# copy all files
COPY requirements.txt /app/

# install dependencies
RUN pip install -r requirements.txt

# copy application code
COPY . /app/

# streamlit port
EXPOSE 8501

# first run the driver to refresh CSVs, then start Streamlit
CMD ["sh", "-c", "python driver.py && \
    streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --browser.serverAddress=localhost"]