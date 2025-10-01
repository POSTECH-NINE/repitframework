# RePIT-Framework

This framework aims at automating the process for ML-CFD cross computation. It is the extension of RePIT algorithm introduced by [J. Jeon et. al](https://arxiv.org/abs/2206.06817). There are basically three ways you can utilize this framework until now:

# Build like a PRO:
You can build this whole like a pro all by yourself. It requires OpenFOAM foundation version pre-installed and good to have full fledged conda setup. 
### Clone the Repository:
Baby step: 
```
git clone git@github.com:JBNU-NINE/repitframework.git
```
### Install OpenFOAM
Don't you worry if you haven't pre-installed OpenFOAM, here are the commands that can lift off your burden:<br>
**Note: put your username in the $USER field --- I know you're a pro, still!**
```
wget -q -O - https://dl.openfoam.org/gpg.key | apt-key add - && \
add-apt-repository http://dl.openfoam.org/ubuntu && \
apt-get update && \
apt-get install -y openfoam12 && \
echo "source /opt/openfoam12/etc/bashrc" >> /home/$USER/.bashrc && \
chown -R $USER:$USER /opt/openfoam12 /home/$USER
```
### conda
I hope you're not one of those who hates conda (well, I haven't found any). And, I really think you've it pre-installed but still if you haven't, I've got your back mate. 
```
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
bash /tmp/miniconda.sh -b -u -p /opt/conda && \
rm /tmp/miniconda.sh

export PATH="/opt/conda/bin:$PATH" >> /home/ninelab/.bashrc

conda init
```
Navigate to this `repitframework` directory and install the required packages `conda env create -f environment.yml` then go to `./repitframework/OpenFOAM/adjustPhiML` directory and do `wmake`. After this you're ready to use the framework to its optimal potential. 
Edit the `config` file, run the `runner.py` file, sit back and relax. 

# Build like a PRO MAX (docker):
### Build and Run container: 
```
docker-compose build
```
If `docker-compose` is not already installed, there is `install_docker_compose.sh` script in the root directory you can install it using: 

```
sudo ./install_docker_compose.sh
```
After the build is completed run `docker-compose up -d`. Please, check on the `docker-compose.yml` file because it is machine specific. For example, if you want to allocate `16GB` or `12CPUs` you have to change the parameters in this file. Also, check if `nvidia-container-toolkit` is pre-installed or not. If it is not installed please follow the steps below: 
### Install the NVIDIA Container Toolkit:
```
# Add NVIDIA's package repository and GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package list and install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```
### Configure Docker to Use the Toolkit:
```
sudo nvidia-ctk runtime configure --runtime=docker
```
### Restart the Docker Daemon (CRITICAL STEP):
```
sudo systemctl restart docker
```
### Verify the Host Setup:
```
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```
### Attach to the Running container:
```
docker exec -it repit_container bash
```
### Start/Stop/Restart the container:
```
docker stop repit_container
docker start repit_container
docker restart repit_container
```

# Build the LAZY (UTLRA PRO MAX) way: 
Pull the fully functional docker image: 
```
docker pull shilaj/repitframework-v1.0:latest
```
```
docker run \
    -d \
    --name repitframework \
    --gpus all \
    -p 8888:8888 \
    -v "/path/on/your/host:/home/ninelab/repitframework" \
    yourusername/repitframework:v1.1-fixed
```
```
docker exec -it repitframework /bin/bash
```