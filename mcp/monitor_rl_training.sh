#!/bin/bash

# Monitor RL Training Script
# Shows real-time training progress from backend logs

echo "🔍 Monitoring RL Training Progress..."
echo "========================================"
echo ""
echo "Looking for training activity in backend logs..."
echo "Press Ctrl+C to stop monitoring"
echo ""

cd "/Users/manavpatel/Documents/API Memory"

# Monitor logs for RL training activity
docker-compose logs -f api 2>&1 | grep --line-buffered -E "train|Training|episode|Episode|reward|Reward|loss|Loss|RL|memory.manager|answer.agent|PPO|trajectory|Trajectory" | while read line; do
    # Add timestamp
    echo "[$(date +'%H:%M:%S')] $line"
done
