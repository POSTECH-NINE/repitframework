# Use the official OpenFOAM 10 image as the base image
FROM openfoam/openfoam11-paraview510

#Set the working directory to the OpenFOAM case directory
WORKDIR /home/openfoam 

USER 0

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev python3-setuptools &&\
    apt-get -y install git htop build-essential software-properties-common && \
    pip3 install pandas numpy matplotlib openturns platypus-opt torch torchvision

#Set the user to the OpenFOAM user:
RUN useradd --user-group --create-home --shell /bin/bash openfoam ;\
    echo "openfoam ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

#Source the user to the OpenFOAM user:
RUN echo ". /opt/openfoam11/etc/bashrc" >> /home/openfoam/.bashrc

#Make the run directory: 
RUN export FOAM_RUN=/home/openfoam/run && mkdir -p $FOAM_RUN 

# COPY the binding script to the run directory: 
COPY ./binding_script.py /home/openfoam/binding_script.py

# Make alias to run the binding script: 
RUN echo "alias Dmake='python3 /home/openfoam/binding_script.py'" >> /home/openfoam/.bashrc
RUN echo "alias python='python3'" >> /home/openfoam/.bashrc
RUN echo "alias home='$HOME'"

#Specify the volume:
VOLUME [ "/home/host_mount" ]

# Expose the port for ParaView/ParaFOAM:
EXPOSE 8080

# Set the DISPLAY environment variable:
ENV DISPLAY=:0

#Run the OpenFOAM solver for your case: 
CMD ["/bin/bash"]

# It is mandatory to install nvidia-container-toolkit in the host machine, to run the container with GPU support. 
# Please refer to this documentation to install it: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker