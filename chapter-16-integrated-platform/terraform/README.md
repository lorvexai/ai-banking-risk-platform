# Phase 1 Shared Infrastructure

Deploys the AWS (eu-west-2 primary) and Azure (uksouth DR) resources all
AWB AI services depend on: VPC + private subnets, the `awb-ai-platform`
ECS cluster, the encrypted PostgreSQL audit database, the platform API
gateway, and the Azure DR backup storage account.

```bash
terraform init
terraform plan  -out=phase1.plan
terraform apply phase1.plan
```

Requires AWS and Azure credentials with the permissions listed in
Section 16.2. Subsequent phases (service task definitions, monitoring,
CRO dashboard) build on these outputs — see ../exercises/deploy_service.sh.
