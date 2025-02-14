#### step 1. build backend serving
cd ray_serve_lnext
# create python env
mkvirtualenv test -p python3.11
pip install -r requirements.txt

# run to save the model to local disk
python model_dl_quant.py # you should see llava_quant folder created

# create docker file Dockerfile

# build docker image
docker build -t ray_serve_lnext .


### step 2. create front end application
mkdir -p frontend
cd frontend
# create app.py and requirements.txt

# build docker image
docker build -t llava_frontend .

### step 3. create docker compose file
# create docker compose file
docker compose up -d
# make sure you open TCP port 8000, 8501 and 8265 on EC2 security group inbound rules
# go to EC2-public-IPV4:8501 to see the frontend UI

# shut down cluster
docker compose down