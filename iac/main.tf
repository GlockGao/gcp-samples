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

  # Firewall configuration
  create_firewall_rules = each.value.create_firewall_rules != null ? each.value.create_firewall_rules : true

  # Pass the list of subnets
  subnets = each.value.subnets
}

# Create VM instance with multiple NICs
module "vm_instance" {
  source = "./instance"

  project_id    = var.project_id
  region        = var.region
  zone          = var.zone
  instance_name = var.instance_name
  machine_type  = var.machine_type

  # Boot disk configuration
  boot_disk_image   = var.boot_disk_image
  boot_disk_size_gb = var.boot_disk_size_gb
  boot_disk_type    = var.boot_disk_type

  # Network interfaces configuration from variables
  network_interfaces = var.network_interfaces

  # Pass VPC network information
  vpc_networks = {
    for name, network in module.vpc_networks : name => {
      network_self_link = network.network_self_link
      subnet_self_links = network.subnet_self_links
    }
  }

  # Instance configuration
  metadata               = var.instance_metadata
  labels                 = var.labels
  tags                   = var.tags
  service_account_email  = ""
  service_account_scopes = ["https://www.googleapis.com/auth/cloud-platform"]

  # Persistent disk configuration
  create_persistent_disk  = var.create_persistent_disk
  persistent_disk_size_gb = var.persistent_disk_size_gb
  persistent_disk_type    = var.persistent_disk_type

  # Scheduling options
  preemptible         = var.preemptible
  automatic_restart   = var.automatic_restart
  on_host_maintenance = var.on_host_maintenance
}
