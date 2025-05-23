# Default values for Terraform variables

# Terraform state
terraform_state_bucket = "easongy-terraform-bucket"

# Project information
project_id = "ali-icbu-gpu-project"
region     = "us-central1"
zone       = "us-central1-a"

# Network configuration
network_name = "terraform-network"
subnet_name  = "terraform-subnet"
subnet_cidr  = "10.0.0.0/24"

# Instance configuration
instance_name     = "terraform-instance"
machine_type      = "e2-medium"
boot_disk_image   = "debian-cloud/debian-11"
boot_disk_size_gb = 20
boot_disk_type    = "pd-standard"

# Tags for the instance
tags = ["http-server", "https-server", "ssh-server"]

# Labels for the instance
labels = {
  environment = "dev"
  managed_by  = "terraform"
  project     = "gcp-samples"
}

# Optional: Create a persistent disk
create_persistent_disk  = false
persistent_disk_size_gb = 100
persistent_disk_type    = "pd-standard"

# Optional: Instance scheduling options
preemptible         = false
automatic_restart   = true
on_host_maintenance = "MIGRATE"
