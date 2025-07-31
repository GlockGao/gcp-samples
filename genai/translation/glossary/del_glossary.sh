#!/bin/bash


# 1. Delete glossary
curl -X DELETE \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "x-goog-user-project: ali-icbu-gpu-project" \
    "https://translation.googleapis.com/v3/projects/ali-icbu-gpu-project/locations/us-central1/glossaries/aligame-glossary"