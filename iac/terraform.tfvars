# Default values for Terraform variables

# Terraform state
terraform_state_bucket = "easongy-terraform-bucket"

# Project information
project_id = "ali-icbu-gpu-project"
region     = "us-central1"
zone       = "us-central1-a"

# Multiple VPC networks configuration with multiple subnets
vpc_networks = {
  # Product VPC with 4 subnets
  "vpc-product" = {
    network_name                 = "vpc-product-t06-gcp-us-central1"
    region                       = "us-central1"
    network_mtu                  = 1500 # High MTU for better performance
    routing_mode                 = "GLOBAL"
    bgp_inter_region_cost        = "DEFAULT"
    bgp_best_path_selection_mode = "STANDARD"
    create_firewall_rules        = true # Enable firewall rules for product VPC
    subnets = [
      {
        name          = "vsw-base-t06-gcp-us-central1-b-01"
        ip_cidr_range = "10.24.180.0/22"
        region        = "us-central1"
        description   = "Product base subnet in us-central1"
      },
      {
        name          = "vsw-ecs-t06-gcp-us-central1-b-01"
        ip_cidr_range = "10.24.160.0/20"
        region        = "us-central1"
        description   = "Product ECS subnet in us-central1"
      },
      {
        name          = "vsw-pod-t06-gcp-us-central1-b-01"
        ip_cidr_range = "10.24.128.0/19"
        region        = "us-central1"
        description   = "Product pod subnet in us-central1"
      },
      {
        name          = "vsw-vip-t06-gcp-us-central1-b-01"
        ip_cidr_range = "10.24.176.0/22"
        region        = "us-central1"
        description   = "Product vip subnet in us-central1"
      },
    ]
  },

  # Storage VPC with 1 subnet
  "vpc-storage" = {
    network_name                 = "vpc-stroage-t06-gcp-us-central1"
    region                       = "us-central1"
    network_mtu                  = 1500
    routing_mode                 = "GLOBAL"
    bgp_inter_region_cost        = "DEFAULT"
    bgp_best_path_selection_mode = "STANDARD"
    create_firewall_rules        = false # Disable firewall rules for storage VPC
    subnets = [
      {
        name          = "vsw-stroage-t06-gcp-us-central1-b-01"
        ip_cidr_range = "10.24.192.0/19"
        region        = "us-central1"
        description   = "Product base subnet in us-central1"
      }
    ]
  },

  # RDMA VPC with RoCE network profile for high-performance computing
  "vpc-rdma" = {
    network_name                 = "vpc-rdma-t06-gcp-us-central1"
    region                       = "us-central1"
    network_mtu                  = 8896 # Maximum MTU for RDMA performance
    routing_mode                 = "REGIONAL"
    bgp_inter_region_cost        = "DEFAULT"
    bgp_best_path_selection_mode = "STANDARD"
    network_profile              = "projects/ali-icbu-gpu-project/global/networkProfiles/us-central1-b-vpc-roce" # Enable RDMA RoCE support
    create_firewall_rules        = false                                                                         # Disable firewall rules for RDMA VPC
    subnets = [
      {
        name          = "vsw-rdma-t06-gcp-us-central1-b-00"
        ip_cidr_range = "10.24.224.0/24"
        region        = "us-central1"
        description   = "RDMA subnet 00 for high-performance workloads"
      },
      {
        name          = "vsw-rdma-t06-gcp-us-central1-b-01"
        ip_cidr_range = "10.24.225.0/24"
        region        = "us-central1"
        description   = "RDMA subnet 01 for high-performance workloads"
      },
      {
        name          = "vsw-rdma-t06-gcp-us-central1-b-02"
        ip_cidr_range = "10.24.226.0/24"
        region        = "us-central1"
        description   = "RDMA subnet 02 for high-performance workloads"
      },
      {
        name          = "vsw-rdma-t06-gcp-us-central1-b-03"
        ip_cidr_range = "10.24.227.0/24"
        region        = "us-central1"
        description   = "RDMA subnet 03 for high-performance workloads"
      },
      {
        name          = "vsw-rdma-t06-gcp-us-central1-b-04"
        ip_cidr_range = "10.24.228.0/24"
        region        = "us-central1"
        description   = "RDMA subnet 04 for high-performance workloads"
      },
      {
        name          = "vsw-rdma-t06-gcp-us-central1-b-05"
        ip_cidr_range = "10.24.229.0/24"
        region        = "us-central1"
        description   = "RDMA subnet 05 for high-performance workloads"
      },
      {
        name          = "vsw-rdma-t06-gcp-us-central1-b-06"
        ip_cidr_range = "10.24.230.0/24"
        region        = "us-central1"
        description   = "RDMA subnet 06 for high-performance workloads"
      },
      {
        name          = "vsw-rdma-t06-gcp-us-central1-b-07"
        ip_cidr_range = "10.24.231.0/24"
        region        = "us-central1"
        description   = "RDMA subnet 07 for high-performance workloads"
      }
    ]
  }
}

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
  goog-ec-src = "vm_add-tf"
}

# Optional: Create a persistent disk
create_persistent_disk  = false
persistent_disk_size_gb = 100
persistent_disk_type    = "pd-standard"

# Optional: Instance scheduling options
preemptible         = false
automatic_restart   = true
on_host_maintenance = "MIGRATE"

# VM instance network interfaces configuration (10 NICs)
network_interfaces = [
  # NIC 1: vpc-product with external IP
  {
    network_name  = "vpc-product"
    subnet_name   = "vsw-base-t06-gcp-us-central1-b-01"
    access_config = true  # External IP for first interface
    nic_type      = "GVNIC"
    queue_count   = 0     # Number of queues for better performance
    stack_type    = "IPV4_ONLY"
  },
  # NIC 2: vpc-storage
  {
    network_name  = "vpc-storage"
    subnet_name   = "vsw-stroage-t06-gcp-us-central1-b-01"
    access_config = false
    nic_type      = "GVNIC"
    queue_count   = 0     # Fewer queues for storage traffic
    stack_type    = "IPV4_ONLY"
  },
  # NIC 3-10: vpc-rdma with MRDMA type for high-performance RDMA workloads
  # {
  #   network_name  = "vpc-rdma"
  #   subnet_name   = "vsw-rdma-t06-gcp-us-central1-b-00"
  #   access_config = false
  #   nic_type      = "MRDMA"  
  #   queue_count   = 8        # Maximum queues for RDMA performance
  #   stack_type    = "IPV4_ONLY"
  # },
  # {
  #   network_name  = "vpc-rdma"
  #   subnet_name   = "vsw-rdma-t06-gcp-us-central1-b-01"
  #   access_config = false
  #   nic_type      = "MRDMA"
  #   queue_count   = 8
  #   stack_type    = "IPV4_ONLY"
  # },
  # {
  #   network_name  = "vpc-rdma"
  #   subnet_name   = "vsw-rdma-t06-gcp-us-central1-b-02"
  #   access_config = false
  #   nic_type      = "MRDMA"
  #   queue_count   = 8
  #   stack_type    = "IPV4_ONLY"
  # },
  # {
  #   network_name  = "vpc-rdma"
  #   subnet_name   = "vsw-rdma-t06-gcp-us-central1-b-03"
  #   access_config = false
  #   nic_type      = "MRDMA"
  #   queue_count   = 8
  #   stack_type    = "IPV4_ONLY"
  # },
  # {
  #   network_name  = "vpc-rdma"
  #   subnet_name   = "vsw-rdma-t06-gcp-us-central1-b-04"
  #   access_config = false
  #   nic_type      = "MRDMA"
  #   queue_count   = 8
  #   stack_type    = "IPV4_ONLY"
  # },
  # {
  #   network_name  = "vpc-rdma"
  #   subnet_name   = "vsw-rdma-t06-gcp-us-central1-b-05"
  #   access_config = false
  #   nic_type      = "MRDMA"
  #   queue_count   = 8
  #   stack_type    = "IPV4_ONLY"
  # },
  # {
  #   network_name  = "vpc-rdma"
  #   subnet_name   = "vsw-rdma-t06-gcp-us-central1-b-06"
  #   access_config = false
  #   nic_type      = "MRDMA"
  #   queue_count   = 8
  #   stack_type    = "IPV4_ONLY"
  # },
  # {
  #   network_name  = "vpc-rdma"
  #   subnet_name   = "vsw-rdma-t06-gcp-us-central1-b-07"
  #   access_config = false
  #   nic_type      = "MRDMA"
  #   queue_count   = 8
  #   stack_type    = "IPV4_ONLY"
  # }
]

# VM instance metadata configuration
instance_metadata = {
  startup-script = "#!/bin/bash\necho 'VM with 10 NICs started successfully' > /var/log/startup.log\necho 'RDMA interfaces configured' >> /var/log/startup.log"
}
