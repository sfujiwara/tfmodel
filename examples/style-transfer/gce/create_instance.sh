#!/usr/bin/env bash

INSTANCE_NAME="style-transfer"
PROJECT_ID=`gcloud config list project --format "value(core.project)"`

# Create Compute Engine instance
gcloud compute --project ${PROJECT_ID} instances create ${INSTANCE_NAME} \
  --zone "us-central1-b" \
  --machine-type "n1-highcpu-16" \
  --network "default" \
  --maintenance-policy "MIGRATE" \
  --scopes "https://www.googleapis.com/auth/cloud-platform" \
  --tags "tensorboard-server" \
  --image "ubuntu-1604-xenial-v20170330" \
  --image-project "ubuntu-os-cloud" \
  --boot-disk-size "200" \
  --boot-disk-type "pd-standard" \
  --boot-disk-device-name ${INSTANCE_NAME} \
  --metadata-from-file startup-script=startup.sh

# Create firewall rule
gcloud compute --project ${PROJECT_ID} firewall-rules create "default-allow-tensorboard" \
  --allow tcp:6006 \
  --network "default" \
  --source-ranges "0.0.0.0/0" \
  --target-tags "tensorboard-server"
