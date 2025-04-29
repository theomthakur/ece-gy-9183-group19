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
  description = "Suffix for resource names (use net ID)"
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
resource "openstack_networking_port_v2" "minimal_port" {
  name            = "sharednet3-xray-node-${var.suffix}"
  network_id      = data.openstack_networking_network_v2.sharednet3.id
  security_group_ids = [
    data.openstack_networking_secgroup_v2.allow_ssh.id
  ]
}

resource "openstack_compute_instance_v2" "minimal_node" {
  name        = "xray-node-${var.suffix}"
  image_name  = "CC-Ubuntu24.04"
  flavor_name = "m1.medium"  // Using the same VM size as your original code
  key_pair    = var.key

  network {
    port = openstack_networking_port_v2.minimal_port.id
  }

  user_data = <<-EOF
    #! /bin/bash
    sudo echo "127.0.1.1 xray-node-${var.suffix}" >> /etc/hosts
    su cc -c /usr/local/bin/cc-load-public-keys
  EOF
}

resource "openstack_networking_floatingip_v2" "minimal_floating_ip" {
  pool        = "public"
  description = "Xray IP for ${var.suffix}"
  port_id     = openstack_networking_port_v2.minimal_port.id
}

// OUTPUTS
output "floating_ip_out" {
  description = "Floating IP assigned to minimal node"
  value       = openstack_networking_floatingip_v2.minimal_floating_ip.address
}
