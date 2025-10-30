-- Rollback Migration: Remove RL Trajectory Tables
-- Description: Drops trajectories and trajectory_steps tables
-- Date: 2025-01-28

-- Drop tables in reverse order (child tables first)
DROP TABLE IF EXISTS trajectory_steps CASCADE;
DROP TABLE IF EXISTS trajectories CASCADE;

-- Drop any remaining indexes (should be dropped with tables, but just in case)
DROP INDEX IF EXISTS idx_trajectories_user_id;
DROP INDEX IF EXISTS idx_trajectories_collection_id;
DROP INDEX IF EXISTS idx_trajectories_agent_type;
DROP INDEX IF EXISTS idx_trajectories_created_at;
DROP INDEX IF EXISTS idx_trajectories_completed_at;

DROP INDEX IF EXISTS idx_trajectory_steps_trajectory_id;
DROP INDEX IF EXISTS idx_trajectory_steps_step_number;
DROP INDEX IF EXISTS idx_trajectory_steps_created_at;
