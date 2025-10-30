#!/bin/bash
# =============================================================================
# AI Memory API - Service Status Checker
# =============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

check_service() {
    local name=$1
    local check_command=$2
    local port=$3

    printf "%-25s" "$name"

    if eval "$check_command" &>/dev/null; then
        echo -e "${GREEN}✓ RUNNING${NC} (port $port)"
        return 0
    else
        echo -e "${RED}✗ DOWN${NC}"
        return 1
    fi
}

check_docker_service() {
    local name=$1
    local container=$2

    printf "%-25s" "$name"

    if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "running")
        if [ "$status" = "healthy" ]; then
            echo -e "${GREEN}✓ HEALTHY${NC}"
        elif [ "$status" = "running" ]; then
            echo -e "${YELLOW}⚠ RUNNING (no health check)${NC}"
        else
            echo -e "${YELLOW}⚠ RUNNING (status: $status)${NC}"
        fi
        return 0
    else
        echo -e "${RED}✗ NOT RUNNING${NC}"
        return 1
    fi
}

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}     AI MEMORY API - SERVICE STATUS CHECK                  ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${YELLOW}Checking Docker Services...${NC}"
echo "───────────────────────────────────────────────────────────"

check_docker_service "PostgreSQL" "memory-postgres"
check_docker_service "Redis" "memory-redis"
check_docker_service "Milvus" "memory-milvus"
check_docker_service "Neo4j" "memory-neo4j"
check_docker_service "Etcd (Milvus dep)" "memory-etcd"
check_docker_service "MinIO (Milvus dep)" "memory-minio"

echo ""
echo -e "${YELLOW}Checking Service Connectivity...${NC}"
echo "───────────────────────────────────────────────────────────"

check_service "PostgreSQL (5432)" "pg_isready -h localhost -p 5432 -U memory_ai" "5432"
check_service "Redis (6379)" "redis-cli -h localhost -p 6379 ping" "6379"
check_service "Milvus (19530)" "curl -f -s http://localhost:19530/healthz" "19530"
check_service "Neo4j Browser (7474)" "curl -f -s http://localhost:7474" "7474"
check_service "Neo4j Bolt (7687)" "nc -z localhost 7687" "7687"

echo ""
echo -e "${YELLOW}Checking API...${NC}"
echo "───────────────────────────────────────────────────────────"

check_service "API Health (8000)" "curl -f -s http://localhost:8000/health" "8000"
check_service "API Docs (8000/docs)" "curl -f -s http://localhost:8000/docs" "8000"

echo ""
echo -e "${YELLOW}Database Tables...${NC}"
echo "───────────────────────────────────────────────────────────"

if docker exec memory-postgres psql -U memory_ai -d memory_ai_db -c "\dt" &>/dev/null; then
    table_count=$(docker exec memory-postgres psql -U memory_ai -d memory_ai_db -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null | tr -d ' \n')
    echo -e "Database Tables: ${GREEN}$table_count tables found${NC}"

    # Check for critical tables
    printf "%-25s" "  users table"
    if docker exec memory-postgres psql -U memory_ai -d memory_ai_db -c "\dt users" 2>/dev/null | grep -q "users"; then
        echo -e "${GREEN}✓ EXISTS${NC}"
    else
        echo -e "${RED}✗ MISSING${NC}"
    fi

    printf "%-25s" "  collections table"
    if docker exec memory-postgres psql -U memory_ai -d memory_ai_db -c "\dt collections" 2>/dev/null | grep -q "collections"; then
        echo -e "${GREEN}✓ EXISTS${NC}"
    else
        echo -e "${RED}✗ MISSING${NC}"
    fi

    printf "%-25s" "  memories table"
    if docker exec memory-postgres psql -U memory_ai -d memory_ai_db -c "\dt memories" 2>/dev/null | grep -q "memories"; then
        echo -e "${GREEN}✓ EXISTS${NC}"
    else
        echo -e "${RED}✗ MISSING${NC}"
    fi

    printf "%-25s" "  procedures table"
    if docker exec memory-postgres psql -U memory_ai -d memory_ai_db -c "\dt procedures" 2>/dev/null | grep -q "procedures"; then
        echo -e "${GREEN}✓ EXISTS${NC}"
    else
        echo -e "${YELLOW}⚠ MISSING (run migration 001)${NC}"
    fi

    printf "%-25s" "  trajectories table"
    if docker exec memory-postgres psql -U memory_ai -d memory_ai_db -c "\dt trajectories" 2>/dev/null | grep -q "trajectories"; then
        echo -e "${GREEN}✓ EXISTS${NC}"
    else
        echo -e "${YELLOW}⚠ MISSING (run migration 002)${NC}"
    fi
else
    echo -e "${RED}Cannot connect to database${NC}"
fi

echo ""
echo -e "${YELLOW}Optional Services...${NC}"
echo "───────────────────────────────────────────────────────────"

check_docker_service "Kafka" "memory-kafka"
check_docker_service "Prometheus" "memory-prometheus"
check_docker_service "Grafana" "memory-grafana"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Overall status
echo ""
if docker ps --format '{{.Names}}' | grep -q "memory-postgres" && \
   docker ps --format '{{.Names}}' | grep -q "memory-redis" && \
   docker ps --format '{{.Names}}' | grep -q "memory-milvus" && \
   docker ps --format '{{.Names}}' | grep -q "memory-neo4j"; then
    echo -e "${GREEN}✓ All critical services are running${NC}"
    echo -e "${YELLOW}Run './test_api_complete.sh' to test API functionality${NC}"
else
    echo -e "${RED}✗ Some critical services are not running${NC}"
    echo -e "${YELLOW}Start services with: docker-compose up -d${NC}"
fi
echo ""
