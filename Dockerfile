FROM omi-registry.e-technik.uni-ulm.de/baas/platform/benchmark-base:v1.2 as builder-image
# 615-python311-inbenchmark-container 
# as builder-image

RUN apt-get update && apt install -y python3-pip && mkdir /binary
# if we want to build postgres dependencies from scratch
# apt install postgresql-common libpq-dev

# COPY install/requirements_py3.11.txt .
# RUN pip3 install --no-cache-dir -r requirements_py3.11.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip3 install --no-cache-dir -r requirements_py3.11.txt 

WORKDIR /opt/code
COPY . .

ENV PYTHONPATH /opt/code
RUN python3 -m venv --copies --clear /binary/venv && \
	/binary/venv/bin/pip3 install -U pip && \
	/binary/venv/bin/pip3 install -r install/requirements_py3.11.txt

FROM omi-registry.e-technik.uni-ulm.de/baas/platform/benchmark-base:v1.2 as deliver-image

COPY --from=builder-image /opt/code/ /opt/code
COPY --from=builder-image /binary/venv /opt/code/venv

ADD init /opt/init

ENV PYTHONPATH /opt/code

WORKDIR /opt/init
ENTRYPOINT ["./entrypoint"]
