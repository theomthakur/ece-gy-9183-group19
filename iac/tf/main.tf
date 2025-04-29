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
  description = "Suffix for resource names (your project number)"
  type        = string
  nullable = false
}

variable "key" {
  description = "Name of key pair"
  type        = string
  default     = "id_rsa_chameleon"
}

// Modified: Reduced to just one node for simplicity
variable "nodes" {
  type = map(string)
  default = {
    "xray-node" = "192.168.1.11"
  }
}

// DATA SOURCES
data "openstack_networking_network_v2" "sharednet3" {
  name = "sharednet3"
}

data "openstack_networking_subnet_v2" "sharednet3_subnet" {
  name = "sharednet3-subnet"
}

data "openstack_networking_secgroup_v2" "allow_ssh" {
  name = "allow-ssh"
}

data "openstack_networking_secgroup_v2" "allow_9001" {
  name = "allow-9001"
}

data "openstack_networking_secgroup_v2" "allow_8000" {
  name = "allow-8000"
}

data "openstack_networking_secgroup_v2" "allow_8080" {
  name = "allow-8080"
}

data "openstack_networking_secgroup_v2" "allow_8081" {
  name = "allow-8081"
}

data "openstack_networking_secgroup_v2" "allow_http_80" {
  name = "allow-http-80"
}

data "openstack_networking_secgroup_v2" "allow_9090" {
  name = "allow-9090"
}

// MAIN RESOURCES
// Modified: Changed names to reflect X-ray project
resource "openstack_networking_network_v2" "private_net" {
  name                  = "private-net-xray-project${var.suffix}"
  port_security_enabled = false
}

resource "openstack_networking_subnet_v2" "private_subnet" {
  name       = "private-subnet-xray-project${var.suffix}"
  network_id = openstack_networking_network_v2.private_net.id
  cidr       = "192.168.1.0/24"
  no_gateway = true
}

// Creating ports on the private network (kept original structure)
resource "openstack_networking_port_v2" "private_net_ports" {
  for_each              = var.nodes
  name                  = "port-${each.key}-xray-project${var.suffix}"
  network_id            = openstack_networking_network_v2.private_net.id
  port_security_enabled = false

  fixed_ip {
    subnet_id  = openstack_networking_subnet_v2.private_subnet.id
    ip_address = each.value
  }
}

// Creating ports on the shared network (kept original structure)
resource "openstack_networking_port_v2" "sharednet3_ports" {
  for_each   = var.nodes
    name       = "sharednet3-${each.key}-xray-project${var.suffix}"
    network_id = data.openstack_networking_network_v2.sharednet3.id
    security_group_ids = [
      data.openstack_networking_secgroup_v2.allow_ssh.id,
      data.openstack_networking_secgroup_v2.allow_9001.id,
      data.openstack_networking_secgroup_v2.allow_8000.id,
      data.openstack_networking_secgroup_v2.allow_8080.id,
      data.openstack_networking_secgroup_v2.allow_8081.id,
      data.openstack_networking_secgroup_v2.allow_http_80.id,
      data.openstack_networking_secgroup_v2.allow_9090.id
    ]
}

// Create VM instances
resource "openstack_compute_instance_v2" "nodes" {
  for_each = var.nodes

  name        = "${each.key}-xray-project${var.suffix}"
  image_name  = "CC-Ubuntu24.04"
  flavor_name = "m1.medium"
  key_pair    = var.key

  network {
    port = openstack_networking_port_v2.sharednet3_ports[each.key].id
  }

  network {
    port = openstack_networking_port_v2.private_net_ports[each.key].id
  }

  user_data = <<-EOF
    #! /bin/bash
    sudo echo "127.0.1.1 ${each.key}-xray-project${var.suffix}" >> /etc/hosts
    su cc -c /usr/local/bin/cc-load-public-keys
  EOF
}

// Added: Storage volume for X-ray data and models
resource "openstack_blockstorage_volume_v2" "xray_storage" {
  name        = "xray-storage-project${var.suffix}"
  size        = 10
  description = "Storage for X-ray detection system data and models"
}

// Added: Attach volume to VM instance
resource "openstack_compute_volume_attach_v2" "xray_volume_attach" {
  instance_id = openstack_compute_instance_v2.nodes["xray-node"].id
  volume_id   = openstack_blockstorage_volume_v2.xray_storage.id
}

// Create floating IP
resource "openstack_networking_floatingip_v2" "floating_ip" {
  pool        = "public"
  description = "X-ray Detection System IP for project${var.suffix}"
  port_id     = openstack_networking_port_v2.sharednet3_ports["xray-node"].id
}

// OUTPUTS
output "floating_ip_out" {
  description = "Floating IP assigned to X-ray node"
  value       = openstack_networking_floatingip_v2.floating_ip.address
}

// Added: Output for the volume ID
output "storage_volume_id" {
  description = "ID of the storage volume for X-ray data"
  value       = openstack_blockstorage_volume_v2.xray_storage.id
}
