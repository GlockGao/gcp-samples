# Variables for the Google Cloud provider configuration

variable "terraform_state_bucket" {
  description = "The GCS bucket for terraform state"
  type        = string
}

variable "project_id" {
  description = "The ID of the GCP project"
  type        = string
}

variable "region" {
  description = "The region to deploy resources to"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The zone to deploy resources to"
  type        = string
  default     = "us-central1-a"
}

# Network configuration variables
variable "vpc_networks" {
  description = "Map of VPC networks to create, each with its own configuration"
  type = map(object({
    network_name = string
    region       = optional(string)
    network_mtu  = optional(number, 1460) # MTU for the VPC network (default: 1460, max: 8896)
    create_nat   = optional(bool, false)
    nat_name     = optional(string)
    router_name  = optional(string)

    # BGP routing configuration
    routing_mode                 = optional(string, "REGIONAL") # REGIONAL or GLOBAL
    bgp_inter_region_cost        = optional(string, "DEFAULT")  # DEFAULT, ADD_COST_TO_MED
    bgp_best_path_selection_mode = optional(string, "STANDARD") # STANDARD, LEGACY

    # Network profile for RDMA support
    network_profile = optional(string)

    # For backward compatibility
    subnet_name = optional(string)
    subnet_cidr = optional(string)

    # New field for multiple subnets
    subnets = optional(list(object({
      name          = string
      ip_cidr_range = string
      region        = optional(string)
      secondary_ip_ranges = optional(list(object({
        range_name    = string
        ip_cidr_range = string
      })), [])
      private_ip_google_access = optional(bool, true)
      description              = optional(string, "Subnet created with Terraform")
    })), [])
  }))
  default = {}
}

# Instance configuration variables
variable "instance_name" {
  description = "The name of the compute instance"
  type        = string
  default     = "terraform-instance"
}

variable "machine_type" {
  description = "The machine type for the compute instance"
  type        = string
  default     = "e2-medium"
}

variable "boot_disk_image" {
  description = "The image for the boot disk"
  type        = string
  default     = "debian-cloud/debian-11"
}

variable "boot_disk_size_gb" {
  description = "The size of the boot disk in GB"
  type        = number
  default     = 20
}

variable "boot_disk_type" {
  description = "The type of the boot disk"
  type        = string
  default     = "pd-standard"
}

variable "tags" {
  description = "Network tags for the instance"
  type        = list(string)
  default     = ["http-server", "https-server", "ssh-server"]
}

variable "labels" {
  description = "Labels to apply to the instance"
  type        = map(string)
  default = {
    environment = "dev"
    managed_by  = "terraform"
    project     = "gcp-samples"
  }
}

variable "create_persistent_disk" {
  description = "Whether to create a persistent disk"
  type        = bool
  default     = false
}

variable "persistent_disk_size_gb" {
  description = "The size of the persistent disk in GB"
  type        = number
  default     = 100
}

variable "persistent_disk_type" {
  description = "The type of the persistent disk"
  type        = string
  default     = "pd-standard"
}

variable "preemptible" {
  description = "Whether the instance is preemptible"
  type        = bool
  default     = false
}

variable "automatic_restart" {
  description = "Whether the instance should be automatically restarted"
  type        = bool
  default     = true
}

variable "on_host_maintenance" {
  description = "VM instance scheduling behavior when the host machine undergoes maintenance"
  type        = string
  default     = "MIGRATE"
}
