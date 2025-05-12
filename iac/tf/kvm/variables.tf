variable "suffix" {
  description = "Suffix to differentiate resources"
  type        = string
  nullable    = false
}

variable "key" {
  description = "Name of the SSH key pair registered in Chameleon"
  type        = string
  default     = "group19-shared-key"
}

variable "nodes" {
  description = "Map of node names to their private IP addresses"
  type        = map(string)
  default     = {
    "node1" = "192.168.1.11"
    "node2" = "192.168.1.12"
    "node3" = "192.168.1.13"
  }
}
