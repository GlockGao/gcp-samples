# Instance module - outputs

output "instance_id" {
  description = "The ID of the compute instance"
  value       = google_compute_instance.vm_instance.id
}

output "instance_name" {
  description = "The name of the compute instance"
  value       = google_compute_instance.vm_instance.name
}

output "instance_self_link" {
  description = "The self link of the compute instance"
  value       = google_compute_instance.vm_instance.self_link
}

output "network_interfaces" {
  description = "All network interfaces of the compute instance"
  value = [
    for ni in google_compute_instance.vm_instance.network_interface : {
      network_ip  = ni.network_ip
      external_ip = length(ni.access_config) > 0 ? ni.access_config[0].nat_ip : null
      network     = ni.network
      subnetwork  = ni.subnetwork
      nic_type    = ni.nic_type
    }
  ]
}

output "primary_internal_ip" {
  description = "The internal IP of the primary network interface"
  value       = length(google_compute_instance.vm_instance.network_interface) > 0 ? google_compute_instance.vm_instance.network_interface[0].network_ip : null
}

output "primary_external_ip" {
  description = "The external IP of the primary network interface (if exists)"
  value       = length(google_compute_instance.vm_instance.network_interface) > 0 && length(google_compute_instance.vm_instance.network_interface[0].access_config) > 0 ? google_compute_instance.vm_instance.network_interface[0].access_config[0].nat_ip : null
}

output "instance_machine_type" {
  description = "The machine type of the compute instance"
  value       = google_compute_instance.vm_instance.machine_type
}

output "instance_zone" {
  description = "The zone of the compute instance"
  value       = google_compute_instance.vm_instance.zone
}

output "boot_disk" {
  description = "The boot disk of the compute instance"
  value = {
    device_name = google_compute_instance.vm_instance.boot_disk[0].device_name
    disk_size   = google_compute_instance.vm_instance.boot_disk[0].initialize_params[0].size
    disk_type   = google_compute_instance.vm_instance.boot_disk[0].initialize_params[0].type
    disk_image  = google_compute_instance.vm_instance.boot_disk[0].initialize_params[0].image
  }
}

output "additional_disk" {
  description = "The additional disk of the compute instance"
  value = var.create_persistent_disk ? {
    name = google_compute_disk.additional_disk[0].name
    size = google_compute_disk.additional_disk[0].size
    type = google_compute_disk.additional_disk[0].type
  } : null
}
