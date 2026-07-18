#!/usr/bin/env bash
# Exercise 16.1: Deploy AWB Credit AI Service Stack
# Deploys MR-2026-035 and MR-2026-037 as ECS Task
# Definitions in AWS eu-west-2.
#
# Prerequisites:
#   aws cli v2 configured with eu-west-2 profile
#   Docker image built and pushed to ECR
#   PostgreSQL audit DB reachable
#
# Usage:
#   chmod +x deploy_service.sh
#   ./deploy_service.sh [env]    # env = dev|staging|prod
#
# Solution: github.com/lorvenio/
#   ai-banking-risk-platform/chapter-16-integrated-platform/solutions/

set -euo pipefail

ENV="${1:-dev}"
REGION="eu-west-2"
CLUSTER="awb-ai-${ENV}"
ECR_REPO="123456789.dkr.ecr.eu-west-2.amazonaws.com"
LOG_GROUP="/awb/ai/${ENV}"

# ── Config ────────────────────────────────────────
CDA_IMAGE="${ECR_REPO}/awb-cda:latest"
CDA_FAMILY="awb-cda-${ENV}"

CDA_AGENT_IMAGE="${ECR_REPO}/awb-credit-agent:latest"
CDA_AGENT_FAMILY="awb-credit-agent-${ENV}"

API_GW_URL="https://api-gw.awb-ai.internal"

echo "=== AWB Credit AI Service Deploy ==="
echo "Env:     ${ENV}"
echo "Region:  ${REGION}"
echo "Cluster: ${CLUSTER}"

# ── Step 1: Authenticate to ECR ───────────────────
echo "[1/6] Authenticating to ECR..."
aws ecr get-login-password \
  --region "${REGION}" | \
  docker login --username AWS \
  --password-stdin "${ECR_REPO}"

# ── Step 2: Register CDA Task Definition ──────────
echo "[2/6] Registering CDA task definition..."
aws ecs register-task-definition \
  --family "${CDA_FAMILY}" \
  --network-mode awsvpc \
  --requires-compatibilities FARGATE \
  --cpu 1024 \
  --memory 2048 \
  --execution-role-arn \
    "arn:aws:iam::123456789:role/awb-ecs-exec" \
  --task-role-arn \
    "arn:aws:iam::123456789:role/awb-cda-task" \
  --container-definitions "[
    {
      \"name\": \"awb-cda\",
      \"image\": \"${CDA_IMAGE}\",
      \"essential\": true,
      \"portMappings\": [
        {\"containerPort\": 8080, \"protocol\": \"tcp\"}
      ],
      \"environment\": [
        {\"name\": \"MODEL_ID\",
         \"value\": \"MR-2026-035\"},
        {\"name\": \"AWS_REGION\",
         \"value\": \"${REGION}\"},
        {\"name\": \"LOG_LEVEL\",
         \"value\": \"INFO\"}
      ],
      \"secrets\": [
        {\"name\": \"GEMINI_API_KEY\",
         \"valueFrom\": \"/awb/gemini_api_key\"},
        {\"name\": \"DB_URL\",
         \"valueFrom\": \"/awb/${ENV}/db_url\"}
      ],
      \"logConfiguration\": {
        \"logDriver\": \"awslogs\",
        \"options\": {
          \"awslogs-group\": \"${LOG_GROUP}\",
          \"awslogs-region\": \"${REGION}\",
          \"awslogs-stream-prefix\": \"cda\"
        }
      }
    }
  ]" \
  --region "${REGION}" \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text

# ── Step 3: Register Credit Agent Task Def ────────
echo "[3/6] Registering Credit Agent task definition..."
aws ecs register-task-definition \
  --family "${CDA_AGENT_FAMILY}" \
  --network-mode awsvpc \
  --requires-compatibilities FARGATE \
  --cpu 2048 \
  --memory 4096 \
  --execution-role-arn \
    "arn:aws:iam::123456789:role/awb-ecs-exec" \
  --task-role-arn \
    "arn:aws:iam::123456789:role/awb-credit-agent-task"\
  --container-definitions "[
    {
      \"name\": \"awb-credit-agent\",
      \"image\": \"${CDA_AGENT_IMAGE}\",
      \"essential\": true,
      \"portMappings\": [
        {\"containerPort\": 8081, \"protocol\": \"tcp\"}
      ],
      \"environment\": [
        {\"name\": \"MODEL_ID\",
         \"value\": \"MR-2026-037\"},
        {\"name\": \"CDA_URL\",
         \"value\": \"http://awb-cda:8080\"},
        {\"name\": \"MAX_COST_GBP\",
         \"value\": \"5.00\"}
      ],
      \"secrets\": [
        {\"name\": \"GEMINI_API_KEY\",
         \"valueFrom\": \"/awb/gemini_api_key\"},
        {\"name\": \"DB_URL\",
         \"valueFrom\": \"/awb/${ENV}/db_url\"}
      ],
      \"logConfiguration\": {
        \"logDriver\": \"awslogs\",
        \"options\": {
          \"awslogs-group\": \"${LOG_GROUP}\",
          \"awslogs-region\": \"${REGION}\",
          \"awslogs-stream-prefix\": \"credit-agent\"
        }
      }
    }
  ]" \
  --region "${REGION}" \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text

# ── Step 4: Update ECS Services ───────────────────
echo "[4/6] Updating ECS services..."
for SVC in \
  "awb-cda-svc:${CDA_FAMILY}" \
  "awb-credit-agent-svc:${CDA_AGENT_FAMILY}"
do
  SVC_NAME="${SVC%%:*}"
  TASK_DEF="${SVC##*:}"
  echo "  Updating ${SVC_NAME} -> ${TASK_DEF}"
  aws ecs update-service \
    --cluster "${CLUSTER}" \
    --service "${SVC_NAME}" \
    --task-definition "${TASK_DEF}" \
    --force-new-deployment \
    --region "${REGION}" \
    --query 'service.serviceName' \
    --output text
done

# ── Step 5: Wait for stability ────────────────────
echo "[5/6] Waiting for services to stabilise..."
aws ecs wait services-stable \
  --cluster "${CLUSTER}" \
  --services awb-cda-svc awb-credit-agent-svc \
  --region "${REGION}"

# ── Step 6: Smoke tests ───────────────────────────
echo "[6/6] Running smoke tests..."

# Health check
CDA_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" \
  "${API_GW_URL}/cda/health")
if [ "${CDA_HEALTH}" != "200" ]; then
  echo "FAIL: CDA health check returned ${CDA_HEALTH}"
  exit 1
fi
echo "  CDA health: OK (200)"

AGENT_HEALTH=$(curl -s -o /dev/null -w "%{http_code}"\
  "${API_GW_URL}/credit-agent/health")
if [ "${AGENT_HEALTH}" != "200" ]; then
  echo "FAIL: Agent health returned ${AGENT_HEALTH}"
  exit 1
fi
echo "  Credit Agent health: OK (200)"

# JWT auth check (should reject without token)
AUTH_CHK=$(curl -s -o /dev/null -w "%{http_code}" \
  "${API_GW_URL}/cda/analyse")
if [ "${AUTH_CHK}" != "401" ]; then
  echo "FAIL: Expected 401 without JWT, got ${AUTH_CHK}"
  exit 1
fi
echo "  JWT RS256 enforcement: OK (401 without token)"

echo ""
echo "=== Deployment complete ==="
echo "Services are stable and smoke tests passed."
echo "Run integration_tests.py for full validation."
