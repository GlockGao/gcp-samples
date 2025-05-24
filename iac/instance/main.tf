# Instance module - main configuration

# Create a compute instance
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

  # Attach to the specified subnet
  network_interface {
    network    = var.network_self_link
    subnetwork = var.subnet_self_link

    # Uncomment to assign an external IP
    access_config {
      # Ephemeral public IP
    }
  }

  # Add metadata
  metadata = {
    startup-script = var.startup_script
  }

  # Add SSH keys if provided
  metadata = {
    ssh-keys = var.ssh_keys
  }

  # Add labels
  labels = var.labels

  # Add tags for firewall rules
  tags = var.tags

  # Service account
  service_account {
    email  = var.service_account_email
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
