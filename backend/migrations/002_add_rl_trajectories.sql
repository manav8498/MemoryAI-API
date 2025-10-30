-- Migration: Add RL Trajectory Tables
-- Description: Adds trajectories and trajectory_steps tables for reinforcement learning
-- Date: 2025-01-28

-- =============================================================================
-- TRAJECTORIES TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS trajectories (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    collection_id VARCHAR(36) REFERENCES collections(id) ON DELETE CASCADE,

    -- Episode metadata
    agent_type VARCHAR(50) NOT NULL,  -- "memory_manager" or "answer_agent"
    final_reward FLOAT,
    total_steps INTEGER NOT NULL DEFAULT 0,

    -- Context
    extra_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,

    CONSTRAINT trajectories_user_id_idx FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT trajectories_collection_id_idx FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
);

CREATE INDEX idx_trajectories_user_id ON trajectories(user_id);
CREATE INDEX idx_trajectories_collection_id ON trajectories(collection_id);
CREATE INDEX idx_trajectories_agent_type ON trajectories(agent_type);
CREATE INDEX idx_trajectories_created_at ON trajectories(created_at DESC);
CREATE INDEX idx_trajectories_completed_at ON trajectories(completed_at DESC);

-- =============================================================================
-- TRAJECTORY STEPS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS trajectory_steps (
    id VARCHAR(36) PRIMARY KEY,
    trajectory_id VARCHAR(36) NOT NULL REFERENCES trajectories(id) ON DELETE CASCADE,

    -- Step information
    step_number INTEGER NOT NULL,

    -- RL tuple: (state, action, reward, next_state)
    state JSONB NOT NULL,
    action INTEGER NOT NULL,  -- Action index (0-3 for memory operations)
    reward FLOAT,
    next_state JSONB,

    -- Value estimates
    value FLOAT,  -- Value function estimate
    log_prob FLOAT,  -- Action log probability

    -- Additional context
    done INTEGER NOT NULL DEFAULT 0,  -- 1 if terminal state
    info JSONB NOT NULL DEFAULT '{}'::jsonb,  -- Extra information

    -- Timestamp
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT trajectory_steps_trajectory_id_idx FOREIGN KEY (trajectory_id) REFERENCES trajectories(id) ON DELETE CASCADE
);

CREATE INDEX idx_trajectory_steps_trajectory_id ON trajectory_steps(trajectory_id);
CREATE INDEX idx_trajectory_steps_step_number ON trajectory_steps(trajectory_id, step_number);
CREATE INDEX idx_trajectory_steps_created_at ON trajectory_steps(created_at DESC);

-- =============================================================================
-- COMMENTS
-- =============================================================================
COMMENT ON TABLE trajectories IS 'RL trajectories (episodes) for training memory agents';
COMMENT ON TABLE trajectory_steps IS 'Individual steps in RL trajectories with state-action-reward tuples';

COMMENT ON COLUMN trajectories.agent_type IS 'Type of agent: memory_manager or answer_agent';
COMMENT ON COLUMN trajectories.final_reward IS 'Total reward accumulated in this episode';
COMMENT ON COLUMN trajectories.total_steps IS 'Number of steps in this trajectory';

COMMENT ON COLUMN trajectory_steps.state IS 'State representation (embedding + context)';
COMMENT ON COLUMN trajectory_steps.action IS 'Action taken (0=ADD, 1=UPDATE, 2=DELETE, 3=NOOP for memory_manager)';
COMMENT ON COLUMN trajectory_steps.reward IS 'Reward received for this action';
COMMENT ON COLUMN trajectory_steps.value IS 'Value function estimate V(s)';
COMMENT ON COLUMN trajectory_steps.log_prob IS 'Log probability of action π(a|s)';
COMMENT ON COLUMN trajectory_steps.done IS 'Whether this is a terminal state (1) or not (0)';
