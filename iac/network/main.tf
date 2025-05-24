# Network module - main configuration

# Create a VPC network
resource "google_compute_network" "vpc_network" {
  name                         = var.network_name
  auto_create_subnetworks      = false
  project                      = var.project_id
  description                  = "VPC Network created with Terraform"
  mtu                          = var.network_mtu
  routing_mode                 = var.routing_mode
  bgp_inter_region_cost        = var.bgp_inter_region_cost
  bgp_best_path_selection_mode = var.bgp_best_path_selection_mode

  # Set network profile only if specified and not null
  network_profile = var.network_profile != null && var.network_profile != "" ? var.network_profile : null
  
  # Lifecycle rule to handle dependencies properly
  lifecycle {
    create_before_destroy = true
  }
}

# Create subnets in the VPC network
resource "google_compute_subnetwork" "subnets" {
  for_each      = { for i, subnet in var.subnets : subnet.name => subnet }
  name          = each.value.name
  ip_cidr_range = each.value.ip_cidr_range
  region        = each.value.region != null ? each.value.region : var.region
  network       = google_compute_network.vpc_network.id
  project       = var.project_id
  description   = each.value.description

  # Only set private_ip_google_access for non-RDMA networks
  # RDMA networks (with network_profile) don't support this field
  private_ip_google_access = var.network_profile == null ? each.value.private_ip_google_access : null

  # Configure secondary IP ranges if provided
  dynamic "secondary_ip_range" {
    for_each = each.value.secondary_ip_ranges
    content {
      range_name    = secondary_ip_range.value.range_name
      ip_cidr_range = secondary_ip_range.value.ip_cidr_range
    }
  }

  # Lifecycle rule to prevent unnecessary recreation
  # For RDMA networks, we ignore changes to private_ip_google_access
  lifecycle {
    ignore_changes = [private_ip_google_access]
  }
}

# Create a firewall rule to allow SSH, HTTP, and HTTPS traffic (optional)
resource "google_compute_firewall" "allow_ssh_http_https" {
  count   = var.create_firewall_rules ? 1 : 0
  name    = "${var.network_name}-allow-ssh-http-https"
  network = google_compute_network.vpc_network.name
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["22", "80", "443"]
  }

  source_ranges = ["0.0.0.0/0"]
  description   = "Allow SSH, HTTP, and HTTPS traffic"

  # Explicit dependency to ensure proper deletion order
  depends_on = [google_compute_network.vpc_network]
}

# Create a firewall rule to allow internal communication within the VPC (optional)
resource "google_compute_firewall" "allow_internal" {
  count   = var.create_firewall_rules ? 1 : 0
  name    = "${var.network_name}-allow-internal"
  network = google_compute_network.vpc_network.name
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }

  # Use subnet CIDR ranges from created subnets
  source_ranges = [for subnet in var.subnets : subnet.ip_cidr_range]
  description   = "Allow internal communication within the VPC"

  # Explicit dependency to ensure proper deletion order
  depends_on = [google_compute_network.vpc_network]
}
