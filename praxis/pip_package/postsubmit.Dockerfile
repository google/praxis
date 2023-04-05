ARG image_name
ARG base_image="gcr.io/pax-on-cloud-project/${image_name}:latest"
FROM $base_image

RUN rm -rf /praxis
COPY . /praxis
RUN pip3 uninstall -y fiddle
RUN pip3 uninstall -y jax
RUN pip3 install --no-deps -r /praxis/praxis/pip_package/requirements.txt

RUN cd /praxis && bazel build ...

RUN cd /praxis && \
  bazel test \
    --test_output=all \
    --test_verbose_timeout_warnings \
    -- \
    praxis/... \
    -praxis/layers:attentions_test \
    -praxis/layers:convolutions_test \
    -praxis/layers:ctc_objectives_test \
    -praxis/layers:embedding_softmax_test \
    -praxis/layers:models_test \
    -praxis/layers:ngrammer_test \
    -praxis/layers:normalizations_test \
    -praxis/layers:transformer_models_test \
    -praxis/layers:transformers_test

WORKDIR /

CMD ["/bin/bash"]
