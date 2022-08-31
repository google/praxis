ARG image_name
ARG base_image="gcr.io/pax-on-cloud-project/${image_name}:latest"
FROM $base_image

RUN rm -rf /praxis
COPY . /praxis
RUN pip install /praxis/praxis/pip_package
RUN cd /praxis && bazel build ...

WORKDIR /

CMD ["/bin/bash"]
