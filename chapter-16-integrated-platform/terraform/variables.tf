variable "aws_region" {
  description = "Primary AWS region (UK data residency)"
  type        = string
  default     = "eu-west-2"
}

variable "azure_region" {
  description = "Azure DR region (DORA recovery site)"
  type        = string
  default     = "uksouth"
}

variable "vpc_cidr" {
  description = "Platform VPC CIDR"
  type        = string
  default     = "10.40.0.0/16"
}

variable "db_instance_class" {
  description = "Audit PostgreSQL instance class"
  type        = string
  default     = "db.r6g.large"
}
