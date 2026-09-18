#!/usr/bin/env bash
# Build the image, push it to ECR, and create or update the App Runner service.
# Idempotent: safe to run from a laptop or from CI on every deploy.
#
# Required env:
#   AWS_REGION, GEMINI_API_KEY, PINECONE_API_KEY, APP_RUNNER_ECR_ACCESS_ROLE_ARN
# Optional env:
#   ECR_REPO (rag-pipeline-qa), SERVICE_NAME (rag-pipeline-qa),
#   PINECONE_INDEX_NAME (rag-pipeline-qa), IMAGE_TAG (git short sha)
set -euo pipefail

for var in AWS_REGION GEMINI_API_KEY PINECONE_API_KEY APP_RUNNER_ECR_ACCESS_ROLE_ARN; do
  if [ -z "${!var:-}" ]; then
    echo "error: $var is not set" >&2
    exit 1
  fi
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ECR_REPO="${ECR_REPO:-rag-pipeline-qa}"
SERVICE_NAME="${SERVICE_NAME:-rag-pipeline-qa}"
PINECONE_INDEX_NAME="${PINECONE_INDEX_NAME:-rag-pipeline-qa}"
IMAGE_TAG="${IMAGE_TAG:-$(git -C "$ROOT" rev-parse --short HEAD)}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
IMAGE="$REGISTRY/$ECR_REPO:$IMAGE_TAG"

echo "==> Ensuring ECR repository $ECR_REPO exists"
aws ecr describe-repositories --repository-names "$ECR_REPO" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$ECR_REPO" \
       --image-scanning-configuration scanOnPush=true >/dev/null

echo "==> Logging in to $REGISTRY"
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

echo "==> Building $IMAGE"
docker build --platform linux/amd64 -t "$IMAGE" -t "$REGISTRY/$ECR_REPO:latest" "$ROOT"

echo "==> Pushing"
docker push "$IMAGE"
docker push "$REGISTRY/$ECR_REPO:latest"

SOURCE_CONFIG="$(cat <<JSON
{
  "ImageRepository": {
    "ImageIdentifier": "$IMAGE",
    "ImageRepositoryType": "ECR",
    "ImageConfiguration": {
      "Port": "8000",
      "RuntimeEnvironmentVariables": {
        "GEMINI_API_KEY": "$GEMINI_API_KEY",
        "PINECONE_API_KEY": "$PINECONE_API_KEY",
        "PINECONE_INDEX_NAME": "$PINECONE_INDEX_NAME"
      }
    }
  },
  "AutoDeploymentsEnabled": false,
  "AuthenticationConfiguration": {
    "AccessRoleArn": "$APP_RUNNER_ECR_ACCESS_ROLE_ARN"
  }
}
JSON
)"

SERVICE_ARN="$(aws apprunner list-services \
  --query "ServiceSummaryList[?ServiceName=='$SERVICE_NAME'].ServiceArn | [0]" \
  --output text)"

if [ -z "$SERVICE_ARN" ] || [ "$SERVICE_ARN" = "None" ]; then
  echo "==> Creating App Runner service $SERVICE_NAME"
  SERVICE_ARN="$(aws apprunner create-service \
    --service-name "$SERVICE_NAME" \
    --source-configuration "$SOURCE_CONFIG" \
    --instance-configuration '{"Cpu":"1024","Memory":"2048"}' \
    --health-check-configuration '{"Protocol":"HTTP","Path":"/health","Interval":10,"Timeout":5,"HealthyThreshold":1,"UnhealthyThreshold":5}' \
    --query Service.ServiceArn --output text)"
else
  echo "==> Updating App Runner service $SERVICE_NAME"
  aws apprunner update-service \
    --service-arn "$SERVICE_ARN" \
    --source-configuration "$SOURCE_CONFIG" >/dev/null
fi

echo "==> Waiting for service to be RUNNING"
for _ in $(seq 1 60); do
  STATUS="$(aws apprunner describe-service --service-arn "$SERVICE_ARN" \
    --query Service.Status --output text)"
  case "$STATUS" in
    RUNNING) break ;;
    CREATE_FAILED|DELETE_FAILED|DELETED) echo "error: service status $STATUS" >&2; exit 1 ;;
    *) sleep 10 ;;
  esac
done

URL="$(aws apprunner describe-service --service-arn "$SERVICE_ARN" \
  --query Service.ServiceUrl --output text)"
echo "==> Deployed: https://$URL"
curl -fsS "https://$URL/health" && echo
