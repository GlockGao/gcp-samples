#!/bin/bash

# 1. Get operation status
curl -X GET \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    https://translation.googleapis.com/v3/projects/ali-icbu-gpu-project/locations/us-central1/operations/20250804-23081754374127-688ea5de-0000-24a1-8952-14c14ef350d8

# 2. Get glossary
curl -X GET \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "x-goog-user-project: ali-icbu-gpu-project" \
    "https://translation.googleapis.com/v3/projects/ali-icbu-gpu-project/locations/us-central1/glossaries/aligame-glossary/glossaryEntries"