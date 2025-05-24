# Network module - variables

variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "network_name" {
  description = "The name of the VPC network"
  type        = string
  default     = "terraform-network"
}

variable "network_mtu" {
  description = "The mtu of the VPC network"
  type        = number
  default     = 1500
}

# BGP routing configuration
variable "routing_mode" {
  description = "The network routing mode (REGIONAL or GLOBAL)"
  type        = string
  default     = "REGIONAL"
  validation {
    condition     = contains(["REGIONAL", "GLOBAL"], var.routing_mode)
    error_message = "Routing mode must be either REGIONAL or GLOBAL."
  }
}

variable "bgp_inter_region_cost" {
  description = "BGP inter-region cost configuration (DEFAULT or ADD_COST_TO_MED)"
  type        = string
  default     = "DEFAULT"
  validation {
    condition     = contains(["DEFAULT", "ADD_COST_TO_MED"], var.bgp_inter_region_cost)
    error_message = "BGP inter-region cost must be either DEFAULT or ADD_COST_TO_MED."
  }
}

variable "bgp_best_path_selection_mode" {
  description = "BGP best path selection mode (STANDARD or LEGACY)"
  type        = string
  default     = "STANDARD"
  validation {
    condition     = contains(["STANDARD", "LEGACY"], var.bgp_best_path_selection_mode)
    error_message = "BGP best path selection mode must be either STANDARD or LEGACY."
  }
}

# Network profile configuration for RDMA support
variable "network_profile" {
  description = "Network profile URL for the VPC. For RDMA, use the zone-specific URL format"
  type        = string
  default     = null
}

variable "zone" {
  description = "The zone for RDMA network profile (required when using RDMA)"
  type        = string
  default     = null
}

variable "subnets" {
  description = "List of subnet configurations to create in the VPC"
  type = list(object({
    name          = string
    ip_cidr_range = string
    region        = optional(string)
    secondary_ip_ranges = optional(list(object({
      range_name    = string
      ip_cidr_range = string
    })), [])
    private_ip_google_access = optional(bool, true)
    description              = optional(string, "Subnet created with Terraform")
  }))
  default = []
}
