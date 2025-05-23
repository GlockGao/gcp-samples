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
variable "network_name" {
  description = "The name of the VPC network"
  type        = string
  default     = "terraform-network"
}

variable "subnet_name" {
  description = "The name of the subnet"
  type        = string
  default     = "terraform-subnet"
}

variable "subnet_cidr" {
  description = "The CIDR range for the subnet"
  type        = string
  default     = "10.0.0.0/24"
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
