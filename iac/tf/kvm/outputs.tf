output "node_names" {
  value       = { for k, v in openstack_compute_instance_v2.nodes : k => v.name }
  description = "Names of all deployed nodes"
}

output "private_ips" {
  value       = var.nodes
  description = "Private IP addresses of the nodes"
}

output "floating_ip_address" {
  value       = openstack_networking_floatingip_v2.node1_floating_ip.address
  description = "Public IP to reach node1 (main node)"
}

output "ssh_command" {
  value       = "ssh -i ~/.ssh/${var.key} cc@${openstack_networking_floatingip_v2.node1_floating_ip.address}"
  description = "SSH command to connect to node1"
}

output "block_volume_name" {
  value       = openstack_blockstorage_volume_v3.blockstorage_volume.name 
  description = "Name of the persistent block volume"
}

output "object_storage_container" {
  value       = openstack_objectstorage_container_v1.objectstore_container.name
  description = "Name of the object storage container"
}

output "ssh_key_used" {
  value       = var.key
  description = "SSH key name used for this deployment"
}

output "private_network" {
  value       = openstack_networking_network_v2.private_net.name
  description = "Name of the private network for inter-node communication"
}

