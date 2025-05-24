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

# Create multiple VPC networks using the vpc_networks variable
module "vpc_networks" {
  source   = "./network"
  for_each = var.vpc_networks

  project_id   = var.project_id
  region       = each.value.region != null ? each.value.region : var.region
  network_name = each.value.network_name
  network_mtu  = each.value.network_mtu

  # BGP routing configuration
  routing_mode                 = each.value.routing_mode != null ? each.value.routing_mode : "REGIONAL"
  bgp_inter_region_cost        = each.value.bgp_inter_region_cost != null ? each.value.bgp_inter_region_cost : "DEFAULT"
  bgp_best_path_selection_mode = each.value.bgp_best_path_selection_mode != null ? each.value.bgp_best_path_selection_mode : "STANDARD"

  # Network profile for RDMA support
  network_profile = each.value.network_profile != null ? each.value.network_profile : null

  # Pass subnet configuration
  subnet_name = each.value.subnet_name != null ? each.value.subnet_name : null
  subnet_cidr = each.value.subnet_cidr != null ? each.value.subnet_cidr : null

  # Pass the list of subnets
  subnets = each.value.subnets
}
