FROM rayproject/ray:latest-py311-gpu
WORKDIR /home/ray

RUN pip3 install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

#note that this is purely for speed of example deplyment
#baking a model in violates dockers no unecessary bloat at build paradigm
#we'll see the use of a networked store as a solution shortly!
COPY ./llava_quant /home/ray/lava_quant/
COPY app.py /home/ray/app.py
COPY constants.py /home/ray/constants.py

#servev
EXPOSE 8000
#dashboard
EXPOSE 8265

# Define the command to run when the container starts
CMD ["sh", "-c", "ray start --head --dashboard-host 0.0.0.0 && serve run app:app"]
