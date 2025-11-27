# /kc:impl Workflow Documentation

## Overview

The `/kc:impl` slash command provides an automated workflow for implementing task specifications. It handles everything from reading the task file to creating a pull request.

This document explains how the workflow works and how to use it effectively.

---

## Command Usage

### Basic Usage

```bash
/kc:impl [task-number]
```

### Examples

```bash
# Implement task 0001 from any feature
/kc:impl 0001

# Implement specific numbered task
/kc:impl 0042
```

### Task Lookup

The command searches for tasks matching the number across all features:
```
docs/specs/*/tasks/{task-number}_*.md
```

If multiple tasks with the same number exist in different features, you'll be prompted to select which one.

---

## Workflow Steps

### 1. Task File Reading

**What Happens:**
- System locates the task file in `docs/specs/*/tasks/`
- Parses the markdown to extract metadata
- Validates required fields are present

**Metadata Extracted:**
- Task number
- Feature name
- Category
- Component(s)
- Agent type
- Estimated size
- Status
- Priority
- Dependencies

**Validation:**
- Task file exists and is readable
- All required metadata fields present
- Task status is not `completed`

### 2. Dependency Check

**What Happens:**
- Reads dependencies from task metadata
- Checks if dependent tasks are completed
- Blocks execution if dependencies not met

**Example:**
```markdown
| **Dependencies** | Task 0001, Task 0003 |
```

System verifies:
- Task 0001 status is `completed`
- Task 0003 status is `completed`

If any dependency is not completed, workflow stops with error message.

### 3. Git Branch Creation

**What Happens:**
- Creates a new git branch with standard naming
- Switches to the new branch
- Confirms branch creation

**Branch Naming Convention:**
```
feature/{feature-name}-{task-number}
```

**Examples:**
```
feature/fix-backend-not-running-on-upload-0001
feature/dark-mode-support-0004
feature/batch-upload-0010
```

**Git Commands Executed:**
```bash
git checkout -b feature/{feature-name}-{task-number}
```

### 4. Agent Selection

**What Happens:**
- Reads `Agent Type` from task metadata
- Routes to appropriate agent

**Agent Routing:**

| Agent Type | Agent Used |
|------------|------------|
| `python-backend-expert` | Implements backend code |
| `frontend-expert` | Implements UI code |
| `code-reviewer` | Reviews implementation |

**Agent Context Provided:**
- Full task file content
- Acceptance criteria list
- Files to modify
- Technical approach
- Testing requirements

### 5. Implementation

**What Happens:**
- Selected agent reads task requirements
- Implements according to acceptance criteria
- Follows technical approach outlined
- Creates/modifies specified files

**Agent Responsibilities:**
- Write clean, well-structured code
- Follow project conventions
- Add appropriate logging
- Handle errors properly
- Follow existing patterns

**Output:**
- New or modified files
- Implementation commits

### 6. Testing

**What Happens:**
- Agent writes unit tests
- Runs test suite to verify
- Ensures all tests pass
- Checks test coverage

**For Python Backend:**
```bash
# Activate environment
cd backend && source .venv/bin/activate

# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov
```

**For Frontend:**
```bash
# Run tests
python -m pytest tests/ -v
```

**Validation:**
- All existing tests still pass (no regressions)
- New tests for new functionality pass
- Coverage meets target (80%+)

### 7. Code Review

**What Happens:**
- `code-reviewer` agent automatically invoked
- Reviews implemented code
- Checks against quality standards
- Validates acceptance criteria met

**Review Checks:**
- Code follows conventions
- Tests are comprehensive
- Documentation updated
- No security issues
- Performance acceptable

**Review Outcomes:**

**If Approved:**
- Proceeds to next step

**If Changes Requested:**
- Agent makes fixes
- Re-runs tests
- Submits for review again
- Loop until approved

### 8. Pull Request Creation

**What Happens:**
- Uses `gh pr create` command
- Generates PR title and description
- Links to task file
- Assigns reviewers if configured

**PR Title Format:**
```
Task {number}: {task title}
```

**PR Description Template:**
```markdown
## Summary
{Task overview}

## Related Task
Task #{task-number}: {task title}
Feature: {feature-name}

## Changes
- Change 1
- Change 2

## Acceptance Criteria
- [x] Criterion 1
- [x] Criterion 2
- [x] All tests pass

## Testing
- [x] Unit tests pass
- [x] Integration tests pass (if applicable)
- [x] Manual testing completed

## Documentation
- [x] Code comments added
- [x] README updated (if needed)
- [x] CHANGELOG updated
```

**GitHub Command:**
```bash
gh pr create \
  --title "Task {number}: {title}" \
  --body "$(cat pr_description.md)" \
  --base main
```

### 9. Task Status Update

**What Happens:**
- Updates task file metadata
- Changes status to `completed`
- Adds Status History entry
- Adds PR information

**Metadata Update:**
```markdown
| **Status** | completed |
```

**Status History Entry:**
```markdown
| 2024-01-16 | completed | PR #123 merged |
```

**PR Information:**
```markdown
| **PR Number** | 123 |
| **Merge Commit** | abc123def456 |
```

**Git Commit:**
```bash
git add docs/specs/{feature-name}/tasks/{task-number}_*.md
git commit -m "Update task {number} status to completed

PR: #123
"
```

---

## Workflow Diagram

```
┌─────────────────────┐
│  /kc:impl {number}  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Read Task File      │
│ Parse Metadata      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Check Dependencies  │
│ All Completed?      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Create Git Branch   │
│ feature/{name}-{#}  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Route to Agent      │
│ Based on Type       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Agent Implements    │
│ Per Acceptance      │
│ Criteria            │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Run Tests           │
│ Verify Coverage     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Code Review         │
│ By Reviewer Agent   │
└──────────┬──────────┘
           │
   ┌───────┴───────┐
   │               │
   ▼               ▼
Approved      Changes Requested
   │               │
   │               └─────────┐
   │                         │
   ▼                         │
┌─────────────────────┐      │
│ Create PR           │      │
│ gh pr create        │      │
└──────────┬──────────┘      │
           │                 │
           ▼                 │
┌─────────────────────┐      │
│ Update Task Status  │      │
│ completed           │      │
└──────────┬──────────┘      │
           │                 │
           ▼                 │
┌─────────────────────┐      │
│ Done! PR #123       │      │
└─────────────────────┘      │
                             │
                ┌────────────┘
                │
                ▼
┌─────────────────────┐
│ Fix Issues          │
│ Re-run Tests        │
└──────────┬──────────┘
           │
           └─────────────────┐
                            │
                            ▼
                    Back to Review
```

---

## Best Practices

### Before Running /kc:impl

1. **Ensure Task is Complete**
   - All sections filled in
   - Acceptance criteria clear
   - Dependencies identified
   - Technical approach documented

2. **Check Dependencies**
   - Verify dependent tasks are completed
   - Ensure dependent code is merged

3. **Review Current Branch**
   - Commit or stash any uncommitted changes
   - Be on main/master branch
   - Pull latest changes

4. **Verify Prerequisites**
   - Tests pass on main branch
   - Build succeeds
   - No known blockers

### During Execution

1. **Monitor Progress**
   - Watch for any errors
   - Review agent decisions
   - Verify implementation matches intent

2. **Validate Tests**
   - Check test coverage
   - Review test quality
   - Ensure edge cases covered

3. **Review Code**
   - Verify follows conventions
   - Check error handling
   - Validate documentation

### After Completion

1. **Review PR**
   - Check PR description
   - Verify all changes intended
   - Review diff for quality

2. **Manual Testing**
   - Test the feature manually
   - Verify acceptance criteria
   - Check for regressions

3. **Merge**
   - Wait for CI/CD if configured
   - Get human review if required
   - Merge when ready

---

## Error Handling

### Common Errors

#### Task Not Found
```
Error: Task 0042 not found in any feature
```

**Solution:**
- Verify task number is correct
- Check task file exists in `docs/specs/*/tasks/`
- Ensure file naming follows convention

#### Dependencies Not Met
```
Error: Task 0003 depends on Task 0001 which is not completed
```

**Solution:**
- Complete dependent tasks first
- Or remove dependency if not needed
- Update task metadata accordingly

#### Branch Already Exists
```
Error: Branch feature/fix-backend-0001 already exists
```

**Solution:**
```bash
# Delete old branch if safe
git branch -D feature/fix-backend-0001

# Or use different branch name by renaming task
```

#### Tests Failing
```
Error: 3 tests failed, cannot proceed
```

**Solution:**
- Fix failing tests
- Ensure implementation is correct
- Check for regressions
- Re-run workflow

#### PR Creation Failed
```
Error: gh pr create failed - authentication required
```

**Solution:**
```bash
# Authenticate with GitHub
gh auth login

# Then re-run /kc:impl
```

---

## Manual Intervention

Sometimes you may need to manually adjust the workflow:

### Pause After Implementation

If you want to review code before creating PR:

1. Let implementation complete
2. Stop before PR creation
3. Review code manually
4. Make adjustments if needed
5. Create PR manually:
   ```bash
   gh pr create --title "Task 0001: Title" --body "Description"
   ```

### Skip Code Review

For trivial changes, you might skip automated review:

**Not Recommended** - Review helps catch issues

### Custom Branch Name

If you need a custom branch name:

1. Create branch manually first
2. Run `/kc:impl` on that branch
3. It will use existing branch

---

## Integration with Git

### Commit Strategy

The workflow creates logical, atomic commits:

1. **Implementation Commit**
   ```
   Implement {feature}

   - Add {functionality}
   - Update {component}

   Task: {number}
   ```

2. **Test Commit**
   ```
   Add tests for {feature}

   - Test {scenario}
   - Cover edge cases

   Task: {number}
   ```

3. **Documentation Commit** (if needed)
   ```
   Update documentation for {feature}

   - Update README
   - Add CHANGELOG entry

   Task: {number}
   ```

### Merge Strategy

Recommended merge strategies:

**Squash and Merge** (Recommended)
- Creates single commit on main
- Clean history
- Easy to revert

**Rebase and Merge**
- Preserves individual commits
- Linear history
- More detailed history

**Merge Commit**
- Preserves all commits
- Shows branch history
- Can be cluttered

---

## Advanced Usage

### Running Multiple Tasks

To implement multiple tasks in sequence:

```bash
/kc:impl 0001
# Wait for completion

/kc:impl 0002
# Wait for completion

/kc:impl 0003
```

### Parallel Implementation

For independent tasks (no dependencies):

Can be run in parallel in different terminal sessions, but ensure they don't modify the same files.

### Re-running Failed Tasks

If a task fails partway through:

1. Fix the issue
2. Re-run `/kc:impl {number}`
3. Workflow resumes from current state

---

## Troubleshooting

### Task Status Stuck

If task status doesn't update:

1. Check task file permissions
2. Verify git can commit changes
3. Manually update status if needed

### Agent Selection Wrong

If wrong agent is selected:

1. Check `Agent Type` in task metadata
2. Update to correct agent type
3. Re-run workflow

### Tests Keep Failing

If tests repeatedly fail:

1. Run tests manually to debug
2. Check test environment
3. Verify dependencies installed
4. Review test logs carefully

---

## Related Documentation

- See `SPECIFICATION_GUIDE.md` for task creation
- See `TASK_TEMPLATE.md` for task structure
- See `AGENT_ROLES.md` for agent details

---

*Last Updated: 2024*
