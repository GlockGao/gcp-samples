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

# Network interface configurations
variable "network_interfaces" {
  description = "List of network interface configurations"
  type = list(object({
    network_name    = string
    subnet_name     = string
    network_ip      = optional(string)
    access_config   = optional(bool, false)
    nic_type        = optional(string, "VIRTIO_NET")  # VIRTIO_NET, GVNIC, or MRDMA
    queue_count     = optional(number)                # Number of queues for the network interface
    stack_type      = optional(string, "IPV4_ONLY")   # IPV4_ONLY, IPV4_IPV6, or IPV6_ONLY
  }))
  default = []
}

variable "vpc_networks" {
  description = "Map of VPC network outputs from network module"
  type = map(object({
    network_self_link = string
    subnet_self_links = map(string)
  }))
  default = {}
}

variable "metadata" {
  description = "Metadata key-value pairs for the compute instance"
  type        = map(string)
  default     = {}
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
