#!/bin/bash

# 1. translate with nmt
# curl -X POST \
#     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
#     -H "x-goog-user-project: ali-icbu-gpu-project" \
#     -H "Content-Type: application/json; charset=utf-8" \
#     -d @glossary.json \
#     "https://translation.googleapis.com/v3/projects/ali-icbu-gpu-project/locations/us-central1/glossaries"