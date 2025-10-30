-- Migration: Add Procedural Memory Tables
-- Description: Adds procedures, procedure_executions, and procedure_templates tables
-- Date: 2025-01-28

-- =============================================================================
-- PROCEDURES TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS procedures (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    collection_id VARCHAR(36) REFERENCES collections(id) ON DELETE CASCADE,

    -- Identification
    name VARCHAR(200) NOT NULL,
    description TEXT,
    category VARCHAR(100),

    -- Structure
    condition JSONB NOT NULL,  -- {"context": "restaurant", "trigger": "bill_received"}
    action JSONB NOT NULL,     -- {"operation": "multiply", "factor": 0.15, "steps": [...]}

    -- Learning metadata
    success_rate FLOAT NOT NULL DEFAULT 0.5,
    usage_count INTEGER NOT NULL DEFAULT 0,
    learned_from JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- Confidence and strength
    confidence FLOAT NOT NULL DEFAULT 0.5,
    strength FLOAT NOT NULL DEFAULT 0.5,

    -- Parameters
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    constraints JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_verified BOOLEAN NOT NULL DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMP,
    last_success_at TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Indexes
    CONSTRAINT procedures_user_id_idx FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT procedures_collection_id_idx FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
);

CREATE INDEX idx_procedures_user_id ON procedures(user_id);
CREATE INDEX idx_procedures_collection_id ON procedures(collection_id);
CREATE INDEX idx_procedures_name ON procedures(name);
CREATE INDEX idx_procedures_category ON procedures(category);
CREATE INDEX idx_procedures_success_rate ON procedures(success_rate DESC);

-- =============================================================================
-- PROCEDURE EXECUTIONS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS procedure_executions (
    id VARCHAR(36) PRIMARY KEY,
    procedure_id VARCHAR(36) NOT NULL REFERENCES procedures(id) ON DELETE CASCADE,

    -- Execution context
    input_state JSONB NOT NULL,
    output_state JSONB,
    parameters_used JSONB NOT NULL,

    -- Outcome
    success BOOLEAN,
    execution_time_ms INTEGER,
    error_message TEXT,

    -- Feedback
    user_feedback FLOAT,  -- -1 to 1
    reward FLOAT,         -- RL reward signal

    -- Timestamps
    executed_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT procedure_executions_procedure_id_idx FOREIGN KEY (procedure_id) REFERENCES procedures(id) ON DELETE CASCADE
);

CREATE INDEX idx_procedure_executions_procedure_id ON procedure_executions(procedure_id);
CREATE INDEX idx_procedure_executions_executed_at ON procedure_executions(executed_at DESC);
CREATE INDEX idx_procedure_executions_success ON procedure_executions(success);

-- =============================================================================
-- PROCEDURE TEMPLATES TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS procedure_templates (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(200) NOT NULL UNIQUE,
    description TEXT,
    category VARCHAR(100),

    -- Template structure
    condition_schema JSONB NOT NULL,
    action_schema JSONB NOT NULL,
    parameter_schema JSONB NOT NULL,

    -- Examples
    examples JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- Metadata
    is_builtin BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_procedure_templates_name ON procedure_templates(name);
CREATE INDEX idx_procedure_templates_category ON procedure_templates(category);
CREATE INDEX idx_procedure_templates_builtin ON procedure_templates(is_builtin);

-- =============================================================================
-- UPDATE TRIGGER FOR procedures.updated_at
-- =============================================================================
CREATE OR REPLACE FUNCTION update_procedures_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_procedures_updated_at
    BEFORE UPDATE ON procedures
    FOR EACH ROW
    EXECUTE FUNCTION update_procedures_updated_at();

-- =============================================================================
-- COMMENTS
-- =============================================================================
COMMENT ON TABLE procedures IS 'Procedural memory: learned procedures and skills (IF-THEN rules)';
COMMENT ON TABLE procedure_executions IS 'Tracks procedure executions for reinforcement learning';
COMMENT ON TABLE procedure_templates IS 'Pre-defined procedure templates for common patterns';

COMMENT ON COLUMN procedures.condition IS 'Condition that triggers this procedure (JSON pattern)';
COMMENT ON COLUMN procedures.action IS 'Action to execute when condition matches';
COMMENT ON COLUMN procedures.success_rate IS 'Historical success rate of this procedure';
COMMENT ON COLUMN procedures.confidence IS 'Current confidence in this procedure (learned via RL)';
COMMENT ON COLUMN procedures.strength IS 'Memory strength (like procedural memory encoding)';
