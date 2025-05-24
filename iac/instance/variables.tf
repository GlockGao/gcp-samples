# Instance module - variables

variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone"
  type        = string
  default     = "us-central1-a"
}

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

variable "network_self_link" {
  description = "The self link of the VPC network"
  type        = string
}

variable "subnet_self_link" {
  description = "The self link of the subnet"
  type        = string
}

variable "startup_script" {
  description = "The startup script for the compute instance"
  type        = string
  default     = ""
}

variable "ssh_keys" {
  description = "The SSH keys for the compute instance"
  type        = string
  default     = ""
}

variable "labels" {
  description = "The labels for the compute instance"
  type        = map(string)
  default     = {}
}

variable "tags" {
  description = "The network tags for the compute instance"
  type        = list(string)
  default     = []
}

variable "service_account_email" {
  description = "The email of the service account for the compute instance"
  type        = string
  default     = ""
}

variable "service_account_scopes" {
  description = "The scopes for the service account"
  type        = list(string)
  default     = ["https://www.googleapis.com/auth/cloud-platform"]
}

variable "create_persistent_disk" {
  description = "Whether to create a persistent disk"
  type        = bool
  default     = false
}

variable "persistent_disk_type" {
  description = "The type of the persistent disk"
  type        = string
  default     = "pd-standard"
}

variable "persistent_disk_size_gb" {
  description = "The size of the persistent disk in GB"
  type        = number
  default     = 100
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
  description = "How to handle host maintenance"
  type        = string
  default     = "MIGRATE"
}
