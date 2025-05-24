# Instance module - main configuration

# Create a compute instance with multiple NICs
resource "google_compute_instance" "vm_instance" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = var.zone
  project      = var.project_id

  boot_disk {
    initialize_params {
      image = var.boot_disk_image
      size  = var.boot_disk_size_gb
      type  = var.boot_disk_type
    }
  }

  # Create multiple network interfaces
  dynamic "network_interface" {
    for_each = var.network_interfaces
    content {
      network     = var.vpc_networks[network_interface.value.network_name].network_self_link
      subnetwork  = var.vpc_networks[network_interface.value.network_name].subnet_self_links[network_interface.value.subnet_name]
      network_ip  = network_interface.value.network_ip
      nic_type    = network_interface.value.nic_type
      queue_count = network_interface.value.queue_count
      stack_type  = network_interface.value.stack_type

      # Add external IP only for the first interface if specified
      dynamic "access_config" {
        for_each = network_interface.value.access_config ? [1] : []
        content {
          # Ephemeral public IP
        }
      }
    }
  }

  # Add metadata including OS Login
  metadata = merge({
    enable-oslogin = "TRUE"
  }, var.metadata)

  # Add labels
  labels = var.labels

  # Add tags for firewall rules
  tags = var.tags

  # Service account
  service_account {
    email  = var.service_account_email != "" ? var.service_account_email : null
    scopes = var.service_account_scopes
  }

  # Allow stopping for update
  allow_stopping_for_update = true

  # Scheduling options
  scheduling {
    preemptible         = var.preemptible
    automatic_restart   = var.automatic_restart
    on_host_maintenance = var.on_host_maintenance
  }
}

# Optionally create and attach a persistent disk
resource "google_compute_disk" "additional_disk" {
  count   = var.create_persistent_disk ? 1 : 0
  name    = "${var.instance_name}-data-disk"
  type    = var.persistent_disk_type
  zone    = var.zone
  size    = var.persistent_disk_size_gb
  project = var.project_id
}

resource "google_compute_attached_disk" "attached_disk" {
  count    = var.create_persistent_disk ? 1 : 0
  disk     = google_compute_disk.additional_disk[0].id
  instance = google_compute_instance.vm_instance.id
  zone     = var.zone
  project  = var.project_id
}
