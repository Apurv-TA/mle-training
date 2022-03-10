FROM continuumio/miniconda3

WORKDIR /usr/local/

# Copy all the files in the repository
COPY . mle_training/

# creating the environment
RUN conda env create -f mle_training/deploy/conda/environment.yml

# After this you can build the container and run the portion of code needed
