FROM debian:bullseye-slim

RUN apt update && apt install python3-pip curl -yy

WORKDIR /home/konnoohmachi

# install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /home/konnoohmachi/rust-install.sh
RUN chmod 755 /home/konnoohmachi/rust-install.sh
RUN /home/konnoohmachi/rust-install.sh -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /home/konnoohmachi
COPY . /home/konnoohmachi

RUN pip3 install virtualenv
RUN virtualenv venv
RUN /bin/bash -c "source venv/bin/activate"

# dev/test dependencies
RUN pip3 install pytest numpy obspy maturin
RUN cargo build
RUN maturin build
RUN pip3 install .
