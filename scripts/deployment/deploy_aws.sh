#!/bin/bash
# ==============================================
# Script de déploiement AWS
# ==============================================

set -e

# Configuration
AWS_REGION=${AWS_REGION:-"eu-west-1"}
ECR_REPOSITORY=${ECR_REPOSITORY:-"cv-classifier"}
ECS_CLUSTER=${ECS_CLUSTER:-"production"}
ECS_SERVICE=${ECS_SERVICE:-"cv-classifier"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Déploiement CV Classifier sur AWS ===${NC}"
echo ""

# Vérifier les prérequis
echo -e "${YELLOW}1. Vérification des prérequis...${NC}"

if ! command -v aws &> /dev/null; then
    echo -e "${RED}AWS CLI non installé. Installer avec: brew install awscli${NC}"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker non installé.${NC}"
    exit 1
fi

# Vérifier la connexion AWS
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}Non connecté à AWS. Exécuter: aws configure${NC}"
    exit 1
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}"

echo -e "${GREEN}✓ Prérequis OK${NC}"
echo "  - AWS Account: ${AWS_ACCOUNT_ID}"
echo "  - Region: ${AWS_REGION}"
echo "  - ECR: ${ECR_URI}"
echo ""

# Connexion à ECR
echo -e "${YELLOW}2. Connexion à ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
echo -e "${GREEN}✓ Connecté à ECR${NC}"
echo ""

# Build de l'image Docker
echo -e "${YELLOW}3. Build de l'image Docker...${NC}"
docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} -f docker/Dockerfile .
echo -e "${GREEN}✓ Image construite${NC}"
echo ""

# Tag et push vers ECR
echo -e "${YELLOW}4. Push vers ECR...${NC}"
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${ECR_URI}:${IMAGE_TAG}
docker push ${ECR_URI}:${IMAGE_TAG}

# Tag latest aussi
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${ECR_URI}:latest
docker push ${ECR_URI}:latest
echo -e "${GREEN}✓ Image pushée vers ECR${NC}"
echo ""

# Déployer sur ECS
echo -e "${YELLOW}5. Déploiement sur ECS...${NC}"
aws ecs update-service \
    --cluster ${ECS_CLUSTER} \
    --service ${ECS_SERVICE} \
    --force-new-deployment \
    --region ${AWS_REGION}

echo -e "${GREEN}✓ Déploiement lancé${NC}"
echo ""

# Attendre le déploiement
echo -e "${YELLOW}6. Attente du déploiement...${NC}"
aws ecs wait services-stable \
    --cluster ${ECS_CLUSTER} \
    --services ${ECS_SERVICE} \
    --region ${AWS_REGION}

echo -e "${GREEN}✓ Déploiement terminé avec succès!${NC}"
echo ""

# Afficher l'URL du load balancer
ALB_DNS=$(aws elbv2 describe-load-balancers \
    --names "${ECS_CLUSTER}-alb" \
    --query 'LoadBalancers[0].DNSName' \
    --output text \
    --region ${AWS_REGION} 2>/dev/null || echo "N/A")

echo "=== Informations de déploiement ==="
echo "  - Image: ${ECR_URI}:${IMAGE_TAG}"
echo "  - Cluster: ${ECS_CLUSTER}"
echo "  - Service: ${ECS_SERVICE}"
echo "  - URL: https://${ALB_DNS}"
echo ""
echo -e "${GREEN}Déploiement réussi!${NC}"
