// TERRAFORM VERSION AND PROVIDER REQUIREMENTS
terraform {
  required_version = ">= 0.14.0"
  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = "~> 1.51.1"
    }
  }
}

// PROVIDER CONFIGURATION
provider "openstack" {
  cloud = "openstack"
}

// VARIABLES
variable "suffix" {
  description = "Suffix for resource names"
  type        = string
  nullable = false
}

variable "key" {
  description = "Name of key pair"
  type        = string
  default     = "id_rsa_chameleon"
}

// DATA SOURCES
data "openstack_networking_network_v2" "sharednet3" {
  name = "sharednet3"
}

data "openstack_networking_secgroup_v2" "allow_ssh" {
  name = "allow-ssh"
}

// MAIN RESOURCES
resource "openstack_networking_port_v2" "main_vm_port" {
  name            = "main-vm-port-${var.suffix}"
  network_id      = data.openstack_networking_network_v2.sharednet3.id
  security_group_ids = [
    data.openstack_networking_secgroup_v2.allow_ssh.id
  ]
}

resource "openstack_compute_instance_v2" "main_vm" {
  name        = "main-vm-${var.suffix}"
  image_name  = "CC-Ubuntu24.04"
  flavor_name = "m1.medium"
  key_pair    = var.key

  network {
    port = openstack_networking_port_v2.main_vm_port.id
  }

  user_data = <<-EOF
    #! /bin/bash
    sudo echo "127.0.1.1 main-vm-${var.suffix}" >> /etc/hosts
    su cc -c /usr/local/bin/cc-load-public-keys
  EOF
}

// CHOOSE ONE OF THE BLOCKS BELOW:

// ====================================================================
// FIRST-TIME RUN - USE THIS BLOCK:
// Uncomment the floating IP resource and comment the rest
// ====================================================================

resource "openstack_networking_floatingip_v2" "main_vm_floating_ip" {
  pool        = "public"
  description = "Floating IP for main-vm-${var.suffix}"
  port_id     = openstack_networking_port_v2.main_vm_port.id
}

// OUTPUTS FOR FIRST-TIME RUN
output "vm_name" {
  description = "Name of the VM"
  value       = openstack_compute_instance_v2.main_vm.name
}

output "network_port_name" {
  description = "Name of the network port"
  value       = openstack_networking_port_v2.main_vm_port.name
}

output "floating_ip_address" {
  description = "Floating IP address"
  value       = openstack_networking_floatingip_v2.main_vm_floating_ip.address
}

output "network_name" {
  description = "Name of connected network"
  value       = data.openstack_networking_network_v2.sharednet3.name
}

output "ssh_command" {
  description = "SSH command to connect to the VM"
  value       = "ssh cc@${openstack_networking_floatingip_v2.main_vm_floating_ip.address}"
}


// ====================================================================
// SUBSEQUENT RUNS - USE THIS BLOCK:
// Keep your floating IP from being destroyed
// ====================================================================
/*
variable "floating_ip_address" {
  description = "Existing floating IP to use"
  default     = "129.114.27.242"  // Update this with your actual IP after first run
}

data "openstack_networking_floatingip_v2" "existing_ip" {
  address = var.floating_ip_address
}

resource "openstack_networking_floatingip_associate_v2" "main_vm_fip_association" {
  floating_ip = data.openstack_networking_floatingip_v2.existing_ip.address
  port_id     = openstack_networking_port_v2.main_vm_port.id
}

// OUTPUTS FOR SUBSEQUENT RUNS
output "vm_name" {
  description = "Name of the VM"
  value       = openstack_compute_instance_v2.main_vm.name
}

output "network_port_name" {
  description = "Name of the network port"
  value       = openstack_networking_port_v2.main_vm_port.name
}

output "floating_ip_address" {
  description = "Floating IP address"
  value       = data.openstack_networking_floatingip_v2.existing_ip.address
}

output "network_name" {
  description = "Name of connected network"
  value       = data.openstack_networking_network_v2.sharednet3.name
}

output "security_group" {
  description = "Security group applied"
  value       = data.openstack_networking_secgroup_v2.allow_ssh.name
}

output "ssh_command" {
  description = "SSH command to connect to the VM"
  value       = "ssh cc@${data.openstack_networking_floatingip_v2.existing_ip.address}"
}
*/
