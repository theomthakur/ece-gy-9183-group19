provider "openstack" {
  cloud = "openstack_kvm"
}

provider "openstack" {
  alias = "swift"
  cloud = "openstack_chi"
}
