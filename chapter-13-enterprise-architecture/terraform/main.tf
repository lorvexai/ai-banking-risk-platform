# terraform/modules/ecs-service/main.tf
# AWB ECS Task Definition — mandatory pattern for all AI services.
# No Kubernetes — ECS Task Definitions at AWB's current scale.
# DORA Art.9 change management | PRA SS1/23 | UK GDPR data residency

# ── Variables ──────────────────────────────────────────────────────
variable "service_name"  { type = string }
variable "image_tag"     { type = string }
variable "ecr_repo"      { type = string }
variable "cpu_units"     { type = number; default = 512 }
variable "memory_mb"     { type = number; default = 1024 }
variable "min_capacity"  { type = number; default = 1 }
variable "max_capacity"  { type = number; default = 10 }
variable "db_secret_arn" { type = string }
variable "jwt_secret_arn"{ type = string }

# ── ECS Task Definition ────────────────────────────────────────────
resource "aws_ecs_task_definition" "ai_service" {
  family                   = var.service_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.cpu_units   # 512–4096
  memory                   = var.memory_mb   # 1024–8192
  execution_role_arn       = aws_iam_role.ecs_exec.arn
  task_role_arn            = aws_iam_role.task_role.arn

  container_definitions = jsonencode([{
    name  = var.service_name
    image = "${var.ecr_repo}:${var.image_tag}"

    portMappings = [{ containerPort = 8000 }]

    environment = [
      { name = "SERVICE_NAME", value = var.service_name },
      { name = "AWS_REGION",   value = "eu-west-2" },
    ]

    # Secrets from AWS Secrets Manager — never in env vars directly
    secrets = [
      { name = "DATABASE_URL",    valueFrom = var.db_secret_arn },
      { name = "JWT_PUBLIC_KEY",  valueFrom = var.jwt_secret_arn },
      { name = "GOOGLE_API_KEY",
        valueFrom = data.aws_secretsmanager_secret.google.arn },
    ]

    healthCheck = {
      command     = ["CMD-SHELL",
        "curl -sf http://localhost:8000/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/awb/${var.service_name}"
        "awslogs-region"        = "eu-west-2"
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])

  tags = {
    Service     = var.service_name
    Environment = "production"
    DataRegion  = "eu-west-2"          # UK GDPR data residency
    DORAAsset   = "true"
  }
}

# ── ECS Service with auto-scaling ─────────────────────────────────
resource "aws_ecs_service" "ai_service" {
  name            = var.service_name
  cluster         = aws_ecs_cluster.domain.id
  task_definition = aws_ecs_task_definition.ai_service.arn
  launch_type     = "FARGATE"

  desired_count                      = var.min_capacity
  deployment_minimum_healthy_percent = 100   # Tier 1 — no downtime
  deployment_maximum_percent         = 200

  network_configuration {
    subnets          = data.aws_subnets.private_app.ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.service.arn
    container_name   = var.service_name
    container_port   = 8000
  }

  service_registries {
    registry_arn = aws_service_discovery_service.service.arn
  }
}

# ── Auto-scaling — CPU > 70% threshold ────────────────────────────
resource "aws_appautoscaling_target" "ecs" {
  max_capacity       = var.max_capacity
  min_capacity       = var.min_capacity
  resource_id        = "service/${aws_ecs_cluster.domain.name}/${var.service_name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "${var.service_name}-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value       = 70.0   # Scale out above 70% CPU
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}

# ── CloudWatch log group — 90-day hot tier ─────────────────────────
resource "aws_cloudwatch_log_group" "service" {
  name              = "/awb/${var.service_name}"
  retention_in_days = 90   # 7-yr S3 Glacier via log archiver
  tags = {
    Service   = var.service_name
    Retention = "7yr-glacier"   # FCA COBS 9
  }
}
