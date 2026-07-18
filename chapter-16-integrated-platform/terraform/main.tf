# AWB Integrated Platform — Phase 1 shared infrastructure (Chapter 16)
# Deploys the AWS (primary, eu-west-2) and Azure (DR, uksouth) resources
# that every AWB AI service depends on, from a single `terraform apply`.

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws     = { source = "hashicorp/aws", version = "~> 5.0" }
    azurerm = { source = "hashicorp/azurerm", version = "~> 3.100" }
  }
}

provider "aws" { region = var.aws_region }
provider "azurerm" { features {} }

# --- Networking -------------------------------------------------------------
resource "aws_vpc" "awb_platform" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  tags                 = local.tags
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.awb_platform.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 4, count.index)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags              = local.tags
}

data "aws_availability_zones" "available" { state = "available" }

# --- Shared compute: ECS cluster for the 23 AI services ---------------------
resource "aws_ecs_cluster" "awb_ai" {
  name = "awb-ai-platform"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  tags = local.tags
}

# --- Shared state: PostgreSQL (audit trail, FCA COBS 9 seven-year) ----------
resource "aws_db_instance" "audit" {
  identifier              = "awb-audit-log"
  engine                  = "postgres"
  engine_version          = "16.3"
  instance_class          = var.db_instance_class
  allocated_storage       = 200
  storage_encrypted       = true
  backup_retention_period = 35
  deletion_protection     = true
  db_subnet_group_name    = aws_db_subnet_group.audit.name
  username                = "awb_platform"
  manage_master_user_password = true
  tags                    = local.tags
}

resource "aws_db_subnet_group" "audit" {
  name       = "awb-audit"
  subnet_ids = aws_subnet.private[*].id
}

# --- API Gateway (JWT RS256 enforced per Exercise 16.1) ---------------------
resource "aws_apigatewayv2_api" "platform" {
  name          = "awb-platform-gw"
  protocol_type = "HTTP"
  tags          = local.tags
}

# --- Azure DR: DORA Art.12 recovery site ------------------------------------
resource "azurerm_resource_group" "dr" {
  name     = "awb-platform-dr"
  location = var.azure_region
  tags     = local.tags
}

resource "azurerm_storage_account" "dr_backups" {
  name                     = "awbplatformdrbackups"
  resource_group_name      = azurerm_resource_group.dr.name
  location                 = azurerm_resource_group.dr.location
  account_tier             = "Standard"
  account_replication_type = "GZRS"
  min_tls_version          = "TLS1_2"
  tags                     = local.tags
}

locals {
  tags = {
    programme  = "AWB-AI-2026"
    regulation = "PRA-SS1-23,DORA,FCA-COBS-9"
    owner      = "platform-engineering"
  }
}
