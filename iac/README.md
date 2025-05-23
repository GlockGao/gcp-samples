# Terraform GCP Infrastructure

This Terraform configuration creates Google Cloud Platform (GCP) infrastructure resources, including a VPC network and compute instances.

## Project Structure

```
iac/
├── main.tf              # Main Terraform configuration file
├── terraform.tfvars     # Default variable values
├── instance/            # Instance module
│   ├── main.tf          # Instance resource definitions
│   ├── variables.tf     # Input variables for the instance module
│   └── outputs.tf       # Output values from the instance module
├── network/             # Network module
│   ├── main.tf          # Network resource definitions
│   ├── variables.tf     # Input variables for the network module
│   └── outputs.tf       # Output values from the network module
└── README.md            # This file
```

## Modules

### Network Module

The network module creates the following resources:
- VPC network
- Subnet
- Firewall rules for SSH, HTTP, HTTPS, and internal communication

### Instance Module

The instance module creates the following resources:
- Compute instance
- Boot disk
- Optional persistent disk

## Prerequisites

- [Terraform](https://www.terraform.io/downloads.html) (v1.0.0 or later)
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- A GCP project with billing enabled
- Appropriate IAM permissions to create resources

## Usage

1. Update the `terraform.tfvars` file with your GCP project ID and other desired values.

2. Initialize Terraform:
   ```
   terraform init
   ```

3. Preview the changes:
   ```
   terraform plan
   ```

4. Apply the changes:
   ```
   terraform apply
   ```

5. To destroy the resources:
   ```
   terraform destroy
   ```

## Customization

You can customize the infrastructure by modifying the variables in the `terraform.tfvars` file or by passing variables on the command line:

```
terraform apply -var="project_id=my-project" -var="instance_name=my-instance"
```

## Variables

### Main Variables

| Name | Description | Default |
|------|-------------|---------|
| project_id | The GCP project ID | (required) |
| region | The GCP region | us-central1 |
| zone | The GCP zone | us-central1-a |
| network_name | The name of the VPC network | terraform-network |
| subnet_name | The name of the subnet | terraform-subnet |
| subnet_cidr | The CIDR range for the subnet | 10.0.0.0/24 |
| instance_name | The name of the compute instance | terraform-instance |
| machine_type | The machine type for the compute instance | e2-medium |

See the `variables.tf` files in each module for additional variables.

## Outputs

### Main Outputs

| Name | Description |
|------|-------------|
| instance_name | The name of the created instance |
| instance_external_ip | The external IP of the created instance |
| network_name | The name of the created VPC network |
| subnet_name | The name of the created subnet |

See the `outputs.tf` files in each module for additional outputs.
