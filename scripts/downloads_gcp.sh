#!/bin/bash
# A Simple Script To Sync results and xBD from Google Cloud
# Chris Requena-Mesa 29/07/2020

echo Authentify with gcloud
gcloud auth login --no-launch-browser

echo Download data in eie-flood-imagery-prediction

#Attention be on the eie_vision directory!
gsutil -m cp -r gs://eie-flood-imagery-prediction/* .