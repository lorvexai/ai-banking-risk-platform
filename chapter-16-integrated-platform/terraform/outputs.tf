output "ecs_cluster_arn" {
  value       = aws_ecs_cluster.awb_ai.arn
  description = "Cluster for all 23 AWB AI service task definitions"
}

output "audit_db_endpoint" {
  value       = aws_db_instance.audit.endpoint
  description = "PostgreSQL audit log (FCA COBS 9 retention)"
}

output "api_gateway_endpoint" {
  value       = aws_apigatewayv2_api.platform.api_endpoint
  description = "Platform API gateway (JWT RS256 enforced)"
}

output "dr_storage_account" {
  value       = azurerm_storage_account.dr_backups.name
  description = "Azure DR backup storage (DORA Art.12)"
}
