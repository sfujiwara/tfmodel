#!/usr/bin/env bash

apt-get update
apt-get -y upgrade

# Install google-fluentd
curl -sSO https://dl.google.com/cloudagents/install-logging-agent.sh
sha256sum install-logging-agent.sh
sudo bash install-logging-agent.sh

# Create config file for google-fluentd
FLUENTD_CONF_FILE="/etc/google-fluentd/config.d/python.conf"
echo "<source>" > ${FLUENTD_CONF_FILE}
echo "  type tail" >> ${FLUENTD_CONF_FILE}
echo "  format json" >> ${FLUENTD_CONF_FILE}
echo "  path /var/log/python/*.log,/var/log/python/*.json" >> ${FLUENTD_CONF_FILE}
echo "  read_from_head true" >> ${FLUENTD_CONF_FILE}
echo "  tag python" >> ${FLUENTD_CONF_FILE}
echo "</source>" >> ${FLUENTD_CONF_FILE}

# Create log directory for Python script
mkdir -p /var/log/python

# Restart google-fluentd
service google-fluentd restart

apt-get -y install python-pip
pip install -U pip
pip install tensorflow
pip install Pillow
pip install scipy

git clone https://github.com/sfujiwara/tfmodel.git
cd tfmodel/examples/style-transfer

tensorboard --logdir=summary &
python style_transfer.py --style img/styles/chojugiga.jpg \
                         --content img/contents/tensorflow.jpg \
                         --tv_weight 0.0 \
                         --content_weight 0.95 \
                         --style_weight=0.05 \
                         --summary_iterations 20 \
                         --iterations 3000 \
                         --learning_rate 1.0