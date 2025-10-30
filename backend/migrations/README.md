# Database Migrations

This directory contains SQL migrations for the AI Memory API database schema.

## Migration Files

### 001_add_procedural_memory.sql
**Purpose:** Adds procedural memory tables for storing learned procedures and skills.

**Tables Added:**
- `procedures`: Stores learned IF-THEN rules and behavioral patterns
- `procedure_executions`: Tracks execution history for reinforcement learning
- `procedure_templates`: Pre-defined procedure templates

**Features:**
- Automatic `updated_at` trigger
- Comprehensive indexes for performance
- JSONB columns for flexible condition/action storage
- RL feedback tracking (rewards, success rates)

### 002_add_rl_trajectories.sql
**Purpose:** Adds RL trajectory tables for reinforcement learning training data.

**Tables Added:**
- `trajectories`: Stores complete RL episodes for both Memory Manager and Answer Agent
- `trajectory_steps`: Individual state-action-reward tuples for each step in episodes

**Features:**
- Support for multiple agent types (memory_manager, answer_agent)
- Complete RL tuple storage (state, action, reward, next_state, done)
- Value function and log probability tracking for PPO training
- Optimized indexes for trajectory retrieval and analysis
- Foreign key cascades for automatic cleanup

## Running Migrations

### Using psql
```bash
# Apply migration
psql -U username -d database_name -f 001_add_procedural_memory.sql

# Rollback migration
psql -U username -d database_name -f 001_add_procedural_memory_rollback.sql
```

### Using Alembic (if configured)
```bash
# Generate migration from models
alembic revision --autogenerate -m "Add procedural memory tables"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1
```

### Using Docker
```bash
# Copy migration into container
docker cp 001_add_procedural_memory.sql api-memory-db:/tmp/

# Execute migration
docker exec -it api-memory-db psql -U postgres -d memory_db -f /tmp/001_add_procedural_memory.sql
```

## Migration Dependencies

**Prerequisites:**
- Existing `users` table
- Existing `collections` table

**Note:** These tables should already exist from the base schema. If not, ensure they are created first.

## Testing Migrations

```sql
-- Verify tables were created
\dt procedures*

-- Check table structure
\d procedures
\d procedure_executions
\d procedure_templates

-- Verify indexes
\di idx_procedures*

-- Test insert
INSERT INTO procedures (id, user_id, name, condition, action)
VALUES (
    gen_random_uuid()::text,
    'test-user-id',
    'test_procedure',
    '{"trigger": "test"}'::jsonb,
    '{"operation": "test"}'::jsonb
);
```

## Schema Changes

### Procedures Table
- Primary key: `id` (VARCHAR(36))
- Foreign keys: `user_id`, `collection_id`
- JSONB fields: `condition`, `action`, `parameters`, `constraints`, `learned_from`
- Performance tracking: `success_rate`, `usage_count`, `confidence`, `strength`

### Procedure Executions Table
- Primary key: `id` (VARCHAR(36))
- Foreign key: `procedure_id`
- JSONB fields: `input_state`, `output_state`, `parameters_used`
- RL fields: `reward`, `user_feedback`, `success`

### Procedure Templates Table
- Primary key: `id` (VARCHAR(36))
- Unique constraint on `name`
- JSONB fields: `condition_schema`, `action_schema`, `parameter_schema`, `examples`

## Rollback Procedure

If issues occur after migration:

1. **Immediate rollback:**
   ```bash
   psql -U username -d database_name -f 001_add_procedural_memory_rollback.sql
   ```

2. **Backup before migration:**
   ```bash
   pg_dump -U username -d database_name > backup_before_migration.sql
   ```

3. **Restore if needed:**
   ```bash
   psql -U username -d database_name < backup_before_migration.sql
   ```

## Migration Checklist

Before running migrations in production:

- [ ] Create database backup
- [ ] Test migration on staging environment
- [ ] Review SQL for potential issues
- [ ] Check for conflicts with existing tables
- [ ] Verify application code is compatible
- [ ] Plan rollback strategy
- [ ] Schedule maintenance window if needed
- [ ] Monitor application logs after migration

## Troubleshooting

### Error: relation "users" does not exist
**Solution:** Ensure base schema is applied first. Users and collections tables must exist.

### Error: permission denied
**Solution:** Ensure database user has CREATE, ALTER, DROP privileges.

### Error: trigger already exists
**Solution:** Safe to ignore if re-running migration. Or use `DROP TRIGGER IF EXISTS` first.

## Future Migrations

When adding new migrations:

1. Increment migration number (002, 003, etc.)
2. Include both upgrade and rollback scripts
3. Document changes in this README
4. Test thoroughly on staging
5. Consider backward compatibility

## Support

For migration issues, check:
- Application logs: `/var/log/api-memory/`
- Database logs: Check PostgreSQL logs
- Schema validation: Run tests in `tests/test_migrations.py`
