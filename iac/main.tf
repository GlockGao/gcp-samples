# Main Terraform configuration file
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "6.36.0"
    }
  }

  backend "gcs" {
    bucket = "easongy-terraform-bucket"
    prefix = "terraform/state"
  }
}

resource "google_storage_bucket" "bucket-for-state" {
  name                        = var.terraform_state_bucket
  location                    = "US"
  uniform_bucket_level_access = true
}

# Configure the Google Cloud provider
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}
