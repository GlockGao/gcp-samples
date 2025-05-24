# Network module - outputs

output "network_id" {
  description = "The ID of the VPC network"
  value       = google_compute_network.vpc_network.id
}

output "network_name" {
  description = "The name of the VPC network"
  value       = google_compute_network.vpc_network.name
}

output "network_self_link" {
  description = "The self link of the VPC network"
  value       = google_compute_network.vpc_network.self_link
}

output "subnets" {
  description = "Map of subnet names to subnet objects"
  value       = google_compute_subnetwork.subnets
}

output "subnet_ids" {
  description = "Map of subnet names to subnet IDs"
  value       = { for name, subnet in google_compute_subnetwork.subnets : name => subnet.id }
}

output "subnet_self_links" {
  description = "Map of subnet names to subnet self links"
  value       = { for name, subnet in google_compute_subnetwork.subnets : name => subnet.self_link }
}

output "subnet_regions" {
  description = "Map of subnet names to subnet regions"
  value       = { for name, subnet in google_compute_subnetwork.subnets : name => subnet.region }
}

output "subnet_cidrs" {
  description = "Map of subnet names to subnet CIDR ranges"
  value       = { for name, subnet in google_compute_subnetwork.subnets : name => subnet.ip_cidr_range }
}

output "firewall_rules" {
  description = "The firewall rules created for the VPC network"
  value = var.create_firewall_rules ? [
    google_compute_firewall.allow_ssh_http_https[0].name,
    google_compute_firewall.allow_internal[0].name
  ] : []
}
