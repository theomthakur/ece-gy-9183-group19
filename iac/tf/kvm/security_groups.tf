# Security Group resources for application ports

# Port 8001
resource "openstack_networking_secgroup_v2" "allow_8001" {
  name        = "allow-8001"
  description = "Allow inbound access on port 8001"
}

resource "openstack_networking_secgroup_rule_v2" "allow_8001" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 8001
  port_range_max    = 8001
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.allow_8001.id
}

# Port 8002
resource "openstack_networking_secgroup_v2" "allow_8002" {
  name        = "allow-8002"
  description = "Allow inbound access on port 8002"
}

resource "openstack_networking_secgroup_rule_v2" "allow_8002" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 8002
  port_range_max    = 8002
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.allow_8002.id
}

# Port 8501
resource "openstack_networking_secgroup_v2" "allow_8501" {
  name        = "allow-8501"
  description = "Allow inbound access on port 8501"
}

resource "openstack_networking_secgroup_rule_v2" "allow_8501" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 8501
  port_range_max    = 8501
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.allow_8501.id
}

# Port 3000
resource "openstack_networking_secgroup_v2" "allow_3000" {
  name        = "allow-3000"
  description = "Allow inbound access on port 3000"
}

resource "openstack_networking_secgroup_rule_v2" "allow_3000" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 3000
  port_range_max    = 3000
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.allow_3000.id
}

# Port 5000
resource "openstack_networking_secgroup_v2" "allow_5000" {
  name        = "allow-5000"
  description = "Allow inbound access on port 5000"
}

resource "openstack_networking_secgroup_rule_v2" "allow_5000" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 5000
  port_range_max    = 5000
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.allow_5000.id
}

# Port 5050 
resource "openstack_networking_secgroup_v2" "allow_5050" {
  name        = "allow-5050"
  description = "Allow inbound access on port 5050"
}

resource "openstack_networking_secgroup_rule_v2" "allow_5050" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 5050
  port_range_max    = 5050
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.allow_5050.id
}

# Port 8888
resource "openstack_networking_secgroup_v2" "allow_8888" {
  name        = "allow-8888"
  description = "Allow inbound access on port 8888"
}

resource "openstack_networking_secgroup_rule_v2" "allow_8888" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 8888
  port_range_max    = 8888
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.allow_8888.id
}

# Port 4000
resource "openstack_networking_secgroup_v2" "allow_4000" {
  name        = "allow-4000"
  description = "Allow inbound access on port 4000"
}

resource "openstack_networking_secgroup_rule_v2" "allow_4000" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 4000
  port_range_max    = 4000
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.allow_4000.id
}

# Port 5432
resource "openstack_networking_secgroup_v2" "allow_5432" {
  name        = "allow-5432"
  description = "Allow inbound access on port 5432"
}

resource "openstack_networking_secgroup_rule_v2" "allow_5432" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 5432
  port_range_max    = 5432
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.allow_5432.id
}

# Port 5051
resource "openstack_networking_secgroup_v2" "allow_5051" {
  name        = "allow-5051"
  description = "Allow inbound access on port 5051"
}

resource "openstack_networking_secgroup_rule_v2" "allow_5051" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 5051
  port_range_max    = 5051
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.allow_5051.id
}

# Port 8793
resource "openstack_networking_secgroup_v2" "allow_8793" {
  name        = "allow-8793"
  description = "Allow inbound access on port 8793"
}

resource "openstack_networking_secgroup_rule_v2" "allow_8793" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 8793
  port_range_max    = 8793
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.allow_8793.id
}

# Port 5555
resource "openstack_networking_secgroup_v2" "allow_5555" {
  name        = "allow-5555"
  description = "Allow inbound access on port 5555"
}

resource "openstack_networking_secgroup_rule_v2" "allow_5555" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 5555
  port_range_max    = 5555
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.allow_5555.id
}

# Port 8265
resource "openstack_networking_secgroup_v2" "allow_8265" {
  name        = "allow-8265"
  description = "Allow inbound access on port 8265"
}

resource "openstack_networking_secgroup_rule_v2" "allow_8265" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 8265
  port_range_max    = 8265
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.allow_8265.id
}

# Port 6379
resource "openstack_networking_secgroup_v2" "allow_6379" {
  name        = "allow-6379"
  description = "Allow inbound access on port 6379"
}

resource "openstack_networking_secgroup_rule_v2" "allow_6379" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 6379
  port_range_max    = 6379
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.allow_6379.id
}

# Port 8090
resource "openstack_networking_secgroup_v2" "allow_8090" {
  name        = "allow-8090"
  description = "Allow inbound access on port 8090"
}

resource "openstack_networking_secgroup_rule_v2" "allow_8090" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 8090
  port_range_max    = 8090
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.allow_8090.id
}

# Additional future ports (7000-7010)
resource "openstack_networking_secgroup_v2" "allow_7000_7010" {
  name        = "allow-7000-7010"
  description = "Allow inbound access on ports 7000-7010 for future use"
}

resource "openstack_networking_secgroup_rule_v2" "allow_7000_7010" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 7000
  port_range_max    = 7010
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.allow_7000_7010.id
}

# Additional future ports (10000-10010)
resource "openstack_networking_secgroup_v2" "allow_10000_10010" {
  name        = "allow-10000-10010"
  description = "Allow inbound access on ports 10000-10010 for future use"
}

resource "openstack_networking_secgroup_rule_v2" "allow_10000_10010" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 10000
  port_range_max    = 10010
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.allow_10000_10010.id
}
