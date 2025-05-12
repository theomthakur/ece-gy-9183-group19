// Private network for inter-node communication
resource "openstack_networking_network_v2" "private_net" {
  name                  = "private-net-group19-${var.suffix}"
  port_security_enabled = false
}

resource "openstack_networking_subnet_v2" "private_subnet" {
  name       = "private-subnet-group19-${var.suffix}"
  network_id = openstack_networking_network_v2.private_net.id
  cidr       = "192.168.1.0/24"
  no_gateway = true
}

resource "openstack_networking_port_v2" "private_net_ports" {
  for_each              = var.nodes
  name                  = "port-${each.key}-group19-${var.suffix}"
  network_id            = openstack_networking_network_v2.private_net.id
  port_security_enabled = false

  fixed_ip {
    subnet_id  = openstack_networking_subnet_v2.private_subnet.id
    ip_address = each.value
  }
}

// Public network ports for each node with security groups
resource "openstack_networking_port_v2" "sharednet2_ports" {
  for_each   = var.nodes
  name       = "sharednet2-${each.key}-group19-${var.suffix}"
  network_id = data.openstack_networking_network_v2.sharednet2.id
  security_group_ids = [
    data.openstack_networking_secgroup_v2.allow_ssh.id,
    data.openstack_networking_secgroup_v2.allow_8000.id,
    data.openstack_networking_secgroup_v2.allow_9000.id,
    data.openstack_networking_secgroup_v2.allow_9001.id,
    data.openstack_networking_secgroup_v2.allow_8080.id,
    data.openstack_networking_secgroup_v2.allow_8081.id,
    data.openstack_networking_secgroup_v2.allow_http_80.id,
    data.openstack_networking_secgroup_v2.allow_9090.id,
    openstack_networking_secgroup_v2.allow_8001.id,
    openstack_networking_secgroup_v2.allow_8002.id,
    openstack_networking_secgroup_v2.allow_8501.id,
    openstack_networking_secgroup_v2.allow_3000.id,
    openstack_networking_secgroup_v2.allow_5000.id,
    openstack_networking_secgroup_v2.allow_5050.id,
    openstack_networking_secgroup_v2.allow_8888.id,
    openstack_networking_secgroup_v2.allow_4000.id,
    openstack_networking_secgroup_v2.allow_5432.id,
    openstack_networking_secgroup_v2.allow_5051.id,
    openstack_networking_secgroup_v2.allow_8793.id,
    openstack_networking_secgroup_v2.allow_5555.id,
    openstack_networking_secgroup_v2.allow_8265.id,
    openstack_networking_secgroup_v2.allow_6379.id,
    openstack_networking_secgroup_v2.allow_8090.id,
    openstack_networking_secgroup_v2.allow_7000_7010.id,
    openstack_networking_secgroup_v2.allow_10000_10010.id
  ]
}

// Node VMs
resource "openstack_compute_instance_v2" "nodes" {
  for_each    = var.nodes
  name        = "${each.key}-group19-${var.suffix}"
  image_name  = "CC-Ubuntu24.04"
  flavor_name = "m1.medium"
  key_pair    = var.key

  // Connect to the shared network
  network {
    port = openstack_networking_port_v2.sharednet2_ports[each.key].id
  }

  // Connect to the private network for inter-node communication
  network {
    port = openstack_networking_port_v2.private_net_ports[each.key].id
  }

  user_data = <<-EOF
    #! /bin/bash
    sudo echo "127.0.1.1 ${each.key}-group19-${var.suffix}" >> /etc/hosts
    su cc -c /usr/local/bin/cc-load-public-keys
  EOF
}

// Assign floating IP to the first node
/* Original floating IP resource that would create a new floating IP
resource "openstack_networking_floatingip_v2" "node1_floating_ip" {
  pool        = "public"
  description = "Floating IP for node1-group19-${var.suffix}"
  port_id     = openstack_networking_port_v2.sharednet2_ports["node1"].id
}
*/

// Create a new floating IP since the old one was destroyed
resource "openstack_networking_floatingip_v2" "node1_floating_ip" {
  pool        = "public"
  description = "Floating IP for node1-group19-${var.suffix}"
  port_id     = openstack_networking_port_v2.sharednet2_ports["node1"].id
}

// Block storage for persistent data (attached to node1)
/* Original block storage resource that would create a new volume
resource "openstack_blockstorage_volume_v3" "blockstorage_volume" {
  name = "blockstorage-volume-${var.suffix}"  // Would create a new volume with dynamic name
  size = 100
  enable_online_resize = true
}
*/

// Keeping the existing block storage resource with lifecycle ignore_changes
resource "openstack_blockstorage_volume_v3" "blockstorage_volume" {
  name = "blockstorage-volume-project19"
  size = 100
  enable_online_resize = true

  lifecycle {
    ignore_changes = all
    prevent_destroy = true
  }
}

// Attach the existing volume to the VM
resource "openstack_compute_volume_attach_v2" "blockstorage_volume_attach" {
  instance_id = openstack_compute_instance_v2.nodes["node1"].id
  volume_id   = openstack_blockstorage_volume_v3.blockstorage_volume.id

  lifecycle {
    ignore_changes = all
  }
}

// Reusing existing object storage container with lifecycle ignore_changes
resource "openstack_objectstorage_container_v1" "objectstore_container" {
  provider = openstack.swift
  name     = "object-persist-project19"

  lifecycle {
    ignore_changes = all
    prevent_destroy = true
  }
}
