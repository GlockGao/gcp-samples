#!/bin/bash

# 1. Get operation status
curl -X GET \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    https://translation.googleapis.com/v3/projects/ali-icbu-gpu-project/locations/us-central1/operations/20250804-00541754294060-688a3846-0000-2598-ac91-14223badc136

# 2. Get glossary
curl -X GET \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "x-goog-user-project: ali-icbu-gpu-project" \
    "https://translation.googleapis.com/v3/projects/ali-icbu-gpu-project/locations/us-central1/glossaries/aligame-glossary/glossaryEntries"