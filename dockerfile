FROM ubuntu:jammy
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git curl
RUN apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
RUN apt install nano

# Install custom python version for use with Julia
WORKDIR /home/active_phase
ENV HOME  /home/active_phase
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl https://pyenv.run | bash
RUN PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.11
RUN pyenv global 3.11

# Install required python packages
RUN pip install scikit-learn scikit-image
RUN pip install paramz six sympy setuptools
RUN pip install numpy==1.23 matplotlib
WORKDIR /GPy
COPY GPy-devel /GPy
RUN python setup.py install
## Install GUI dependency
RUN apt install libgl1-mesa-glx -y
RUN apt install libxcb-cursor0 -y
RUN apt install libdbus-1-dev -y
RUN apt install -y '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
RUN pip install PyQT5
WORKDIR /mayavi
COPY mayavi /mayavi
RUN pip install .

# Install Julia
WORKDIR $HOME
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.3-linux-x86_64.tar.gz
RUN tar zxvf julia-1.9.3-linux-x86_64.tar.gz
ENV PATH $PATH:$HOME/julia-1.9.3/bin

# Integrate Julia with python
RUN julia -e 'using Pkg; Pkg.add("PyCall")'
RUN pip install julia

# Copy my files over
COPY ./AugmentedGaussianProcesses.jl $HOME/AugmentedGaussianProcesses.jl
# Precompile packages. Needs git repo
WORKDIR $HOME/AugmentedGaussianProcesses.jl
RUN  git config --global user.email "you@example.com"
RUN  git config --global user.name "Your Name"
RUN git init
RUN git add .
RUN git commit -m "Initial commit"
RUN julia install.jl


# docker run -it --net=host --entrypoint /bin/bash -v /mnt/storage_ssd/PDSp:/opt/project -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix phase_sample:py311
# docker run -it -e DISPLAY -v /home/maccyz/Documents/active_phase:/opt/project --net host phase_sample:gui bash
