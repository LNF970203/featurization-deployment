FROM python:3.12.3-alpine

WORKDIR /flask

COPY requirements.txt . 

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "python3", "-m" , "flask","--debug", "run", "--host=0.0.0.0"]