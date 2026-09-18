# Deployment

One container (API + built frontend) pushed to **AWS ECR** and run on
**AWS App Runner**. `scripts/deploy.sh` does the whole thing and is what
the `Deploy` GitHub Actions workflow calls, so laptop and CI deploys are
identical.

## Run the container locally

```bash
docker build -t rag-pipeline-qa .
docker run --rm -p 8000:8000 --env-file backend/.env rag-pipeline-qa
```

Then open <http://localhost:8000> (UI) or `GET /health`.

## One-time AWS setup

1. Install the [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
   and run `aws configure` with an IAM user that can manage ECR and App
   Runner (`AmazonEC2ContainerRegistryFullAccess` + `AWSAppRunnerFullAccess`
   is enough for a personal project).
2. Create the role App Runner uses to pull from ECR:

   ```bash
   aws iam create-role --role-name AppRunnerECRAccessRole \
     --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"build.apprunner.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
   aws iam attach-role-policy --role-name AppRunnerECRAccessRole \
     --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
   aws iam get-role --role-name AppRunnerECRAccessRole --query Role.Arn --output text
   ```

   Keep the ARN it prints.

## Deploy from your machine

```bash
export AWS_REGION=us-east-1
export APP_RUNNER_ECR_ACCESS_ROLE_ARN=arn:aws:iam::<account>:role/AppRunnerECRAccessRole
export GEMINI_API_KEY=...
export PINECONE_API_KEY=...
bash scripts/deploy.sh
```

The script creates the ECR repo and App Runner service on first run and
updates them afterwards, waits until the service is `RUNNING`, and prints
the public URL.

## Deploy from GitHub Actions

Repository **secrets**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`APP_RUNNER_ECR_ACCESS_ROLE_ARN`, `GEMINI_API_KEY`, `PINECONE_API_KEY`.

Repository **variables**: `AWS_REGION` (required), `PINECONE_INDEX_NAME`
(optional), `DEPLOY_ON_PUSH` - set to `true` to deploy every push to
`main`; leave unset to deploy only via *Run workflow*.

## Cost

App Runner bills provisioned memory even while idle (roughly USD 5/month
for the 1 vCPU / 2 GB instance configured here) plus per-request compute.
Pause the service from the App Runner console when not demoing:
`aws apprunner pause-service --service-arn <arn>`.

## Corpus

Pinecone is the single source of truth for the corpus: both dense search
and the BM25 keyword index read from it, so documents uploaded through
`POST /documents` (or the UI) survive restarts and redeploys without any
volume. `backend/data/` is only a convenience for bulk-indexing local files
from a laptop.
