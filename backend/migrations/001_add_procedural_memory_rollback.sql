-- Migration Rollback: Remove Procedural Memory Tables
-- Description: Rollback migration 001_add_procedural_memory.sql
-- Date: 2025-01-28

-- Drop triggers first
DROP TRIGGER IF EXISTS trigger_update_procedures_updated_at ON procedures;
DROP FUNCTION IF EXISTS update_procedures_updated_at();

-- Drop tables in reverse order (respect foreign keys)
DROP TABLE IF EXISTS procedure_executions CASCADE;
DROP TABLE IF EXISTS procedure_templates CASCADE;
DROP TABLE IF EXISTS procedures CASCADE;

-- Drop indexes (if tables aren't dropped with CASCADE)
DROP INDEX IF EXISTS idx_procedures_user_id;
DROP INDEX IF EXISTS idx_procedures_collection_id;
DROP INDEX IF EXISTS idx_procedures_name;
DROP INDEX IF EXISTS idx_procedures_category;
DROP INDEX IF EXISTS idx_procedures_success_rate;

DROP INDEX IF EXISTS idx_procedure_executions_procedure_id;
DROP INDEX IF EXISTS idx_procedure_executions_executed_at;
DROP INDEX IF EXISTS idx_procedure_executions_success;

DROP INDEX IF EXISTS idx_procedure_templates_name;
DROP INDEX IF EXISTS idx_procedure_templates_category;
DROP INDEX IF EXISTS idx_procedure_templates_builtin;
