# Task Specification System Guide

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Task File Format](#task-file-format)
4. [Task Numbering and Naming](#task-numbering-and-naming)
5. [Task Sizing Guidelines](#task-sizing-guidelines)
6. [Agent Types](#agent-types)
7. [Task Categories](#task-categories)
8. [Dependency Management](#dependency-management)
9. [Status Management](#status-management)
10. [Git Workflow Integration](#git-workflow-integration)
11. [Best Practices](#best-practices)
12. [Examples](#examples)

---

## Overview

The Task Specification System provides a structured approach to planning, tracking, and implementing features in the Locopilot Desktop Application. It integrates with the `/kc:impl` slash command to enable automated implementation workflows.

### Key Benefits

- **Structured Planning**: Organize features and tasks systematically
- **Clear Tracking**: Monitor progress through task status
- **Automated Workflow**: Integrate with `/kc:impl` for agent-driven implementation
- **Quality Assurance**: Built-in testing and review requirements
- **Documentation**: Maintain comprehensive implementation history

---

## Directory Structure

```
docs/specs/
├── _meta/                              # System documentation
│   ├── SPECIFICATION_GUIDE.md          # This guide
│   ├── TASK_TEMPLATE.md                # Template for new tasks
│   ├── AGENT_ROLES.md                  # Agent type definitions
│   └── WORKFLOW.md                     # /kc:impl workflow documentation
│
└── {feature-name}/                     # Feature directory
    ├── FEATURE_OVERVIEW.md             # Feature description and goals
    ├── ARCHITECTURE.md                 # Design decisions and approach
    ├── REQUIREMENTS.md                 # Requirements and acceptance criteria
    ├── TESTING_STRATEGY.md             # Testing approach
    │
    └── tasks/                          # Task files
        ├── 0001_task_title.md
        ├── 0002_task_title.md
        └── ...
```

### Directory Guidelines

1. **`_meta/`**: Contains system-wide documentation. Do not create feature subdirectories here.
2. **`{feature-name}/`**: One directory per feature/epic. Use kebab-case naming.
3. **`tasks/`**: Contains numbered task files for sequential implementation.

---

## Task File Format

Each task is documented in a markdown file with the following structure:

### Required Sections

1. **Metadata Table**: Structured information for automation
2. **Overview**: 1-2 sentence summary
3. **Description**: Detailed explanation of the work
4. **Acceptance Criteria**: Checklist of completion requirements
5. **Implementation Details**: Files to modify and technical approach
6. **Testing Strategy**: Test coverage requirements
7. **Technical Notes**: Constraints and considerations
8. **Implementation Checklist**: Step-by-step execution plan

### Optional Sections

- **Related Tasks**: Links to dependent or related tasks
- **Documentation Updates**: Required doc changes
- **Status History**: Timeline of status changes
- **PR Information**: Pull request details after creation

### Complete Template

See `TASK_TEMPLATE.md` for the full reusable template.

---

## Task Numbering and Naming

### File Naming Convention

```
{TASK_NUMBER}_{kebab-case-description}.md
```

**Examples:**
```
0001_implement_backend_health_check.md
0002_add_retry_logic_to_upload.md
0003_update_ui_error_messages.md
```

### Task Number Format

- **Format**: Four digits, zero-padded (0001-9999)
- **Range**: 0001 to 9999 per feature
- **Increment**: Sequential within each feature
- **No Reuse**: Once assigned, never reuse a task number

### Naming Best Practices

1. **Use Action Verbs**: Start with implement, add, fix, refactor, update
2. **Be Specific**: Describe what exactly is being done
3. **Keep Concise**: Maximum 5-6 words in the title
4. **Use Kebab-Case**: Lowercase with hyphens between words

**Good Examples:**
- `0001_implement_trip_status_update.md`
- `0002_add_dark_mode_toggle.md`
- `0003_fix_backend_startup_failure.md`

**Bad Examples:**
- `0001_stuff.md` (too vague)
- `0002_implement_new_feature_for_users.md` (too long)
- `0003_Fix_Backend.md` (wrong case)

---

## Task Sizing Guidelines

Tasks are sized to help estimate complexity and time required.

### Size Categories

| Size | Duration | Complexity | Description |
|------|----------|------------|-------------|
| **S** (Small) | 1-3 hours | Low | Simple changes, small utilities, documentation |
| **M** (Medium) | 3-8 hours | Medium | New service methods, UI components, test suites |
| **L** (Large) | 1-3 days | High | Major features, complex services, refactoring |
| **XL** (Extra Large) | 3+ days | Very High | Architectural changes, complete subsystems |

### Sizing Guidelines

1. **Single Responsibility**: Each task should handle one logical unit of work
2. **Target Medium**: Aim for Medium-sized tasks for optimal workflow
3. **Split Large Tasks**: If a task is XL, consider breaking into multiple L or M tasks
4. **Independence**: Tasks should be completable independently (with dependencies noted)

### Examples by Size

**Small (S):**
- Add a new enum value
- Update docstrings
- Fix typo in error message
- Add logging to existing function

**Medium (M):**
- Implement new service method
- Create UI widget
- Add unit test suite for module
- Refactor function for clarity

**Large (L):**
- Implement new feature subsystem
- Major refactoring of service layer
- Add integration tests across modules
- Implement complex business logic

**Extra Large (XL):**
- Complete backend rewrite
- Migrate to new framework
- Implement real-time sync system
- Major architectural redesign

---

## Agent Types

Tasks are categorized by agent type to enable automated routing in the `/kc:impl` workflow.

### python-backend-expert

**Suitable for:**
- Service layer implementation
- Backend models (Pydantic)
- API integration
- Utility functions
- Configuration management
- Backend business logic

**Typical Files:**
- `services/*.py`
- `models/*.py`
- `utils/*.py`
- `tests/test_services.py`
- `tests/test_models.py`

### frontend-expert

**Suitable for:**
- UI/UX implementation
- View creation and modification
- Widget development
- Event handling
- User interactions
- PySide6 components

**Typical Files:**
- `views/*.py`
- `views/widgets/*.py`
- `controllers/*.py`
- `resources/*`

### code-reviewer

**Suitable for:**
- Code review automation
- Quality checks
- Test verification
- Documentation review
- Best practices enforcement

**Usage:** Automatically invoked after implementation to review changes.

See `AGENT_ROLES.md` for detailed agent responsibilities.

---

## Task Categories

Tasks are categorized by the project component they modify:

| Category | Description | Example Tasks |
|----------|-------------|---------------|
| **models** | Data models and validation | Add TripStatus enum, create new Pydantic model |
| **views** | UI components and layouts | Create LoginView, add dark mode toggle |
| **controllers** | Business logic flow | Implement TripsController, add error handling |
| **services** | External integrations | Implement AuthService, add S3 upload |
| **utils** | Shared utilities | Add logger, create config helper |
| **tests** | Test coverage | Add unit tests, create integration tests |
| **build** | Build and deployment | Update PyInstaller spec, add CI/CD step |
| **docs** | Documentation | Update README, add API docs |

---

## Dependency Management

### Declaring Dependencies

In the task metadata table:

```markdown
| **Dependencies** | Task 0001, Task 0003 |
```

Or if no dependencies:

```markdown
| **Dependencies** | None |
```

### Dependency Rules

1. **Minimize Chains**: Avoid long dependency chains (max 3-4 deep)
2. **Parallel Execution**: Group independent tasks together
3. **Explain Why**: Always document why a dependency exists
4. **Critical Path**: Identify tasks that block others

### Dependency Types

**Sequential Dependencies:**
```
Task 0001: Create Service Class
    ↓
Task 0002: Implement Service Methods
    ↓
Task 0003: Add Service to Controller
```

**Parallel Tasks (no dependencies):**
```
Task 0001: Create LoginView
Task 0002: Create TripsView
Task 0003: Create SettingsView
```

**Complex Dependencies:**
```
Task 0001: Create ThemeManager
    ├─→ Task 0002: Implement Dark Mode
    └─→ Task 0003: Implement Light Mode
         ↓
Task 0004: Add Theme Toggle
```

### Handling Blocked Tasks

If a task becomes blocked:

1. Update status to `blocked`
2. Add entry to Status History explaining the blocker
3. Create a new task to resolve the blocker if needed
4. Communicate the blocker to the team

---

## Status Management

### Valid Status Values

| Status | Meaning | When to Use |
|--------|---------|-------------|
| **pending** | Not started | Initial state when task is created |
| **in_progress** | Active work | Someone is actively implementing |
| **completed** | Finished | All acceptance criteria met, PR merged |
| **blocked** | Cannot proceed | Waiting on dependency or external factor |

### Status Transitions

```
pending → in_progress → completed
   ↓           ↓
   └─→ blocked → (resolve blocker) → in_progress → completed
```

### Updating Status

**When starting work:**
1. Update metadata: `| **Status** | in_progress |`
2. Add Status History entry:
```markdown
| 2024-01-15 | in_progress | Started implementation |
```

**When completing:**
1. Verify all acceptance criteria are met
2. Update metadata: `| **Status** | completed |`
3. Add Status History entry with PR number:
```markdown
| 2024-01-16 | completed | PR #123 merged |
```

**When blocked:**
1. Update metadata: `| **Status** | blocked |`
2. Add Status History entry with blocker details:
```markdown
| 2024-01-15 | blocked | Waiting for API endpoint deployment |
```

---

## Git Workflow Integration

### Branch Naming Convention

```
feature/{feature-name}-{task-number}
```

**Examples:**
```
feature/fix-backend-not-running-on-upload-0001
feature/dark-mode-support-0004
feature/trip-status-update-0002
```

### Commit Message Guidelines

1. **Use Present Tense**: "Add feature" not "Added feature"
2. **Be Specific**: Explain what and why
3. **Reference Task**: Include task number in footer
4. **Follow Convention**: Use project's commit message style

**Example:**
```
Implement backend health check endpoint

Add /health endpoint to verify backend is running and
responsive. Returns 200 OK with status information.

Task: 0001
```

### Pull Request Template

When creating PRs via `/kc:impl`, include:

```markdown
## Summary
Brief description of changes

## Related Task
Task #0001: {task title}

## Changes
- Change 1
- Change 2

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Documentation
- [ ] Code comments added
- [ ] README updated if needed
```

---

## Best Practices

### Writing Effective Tasks

#### 1. Clear Titles
Use action verbs and be specific:
- ✅ Good: "Implement user authentication service"
- ❌ Bad: "Authentication stuff"

#### 2. Measurable Acceptance Criteria
Make criteria specific and testable:
- ✅ Good: "Login endpoint returns JWT token within 500ms"
- ❌ Bad: "Login works"

#### 3. Link to Existing Patterns
Reference code patterns to follow:
```markdown
Follow the pattern in `controllers/login_controller.py` for
worker thread signal handling.
```

#### 4. Document Edge Cases
Include known edge cases:
- Handle case where video file is corrupted
- Handle case where S3 upload fails midway
- Handle case where network connection drops

#### 5. Realistic Time Estimates
Size tasks appropriately:
- Small: 1-3 hours
- Medium: 3-8 hours (target)
- Large: 1-3 days
- XL: Split into smaller tasks

### Code Quality Standards

Every task must meet these standards:

- [ ] All acceptance criteria met
- [ ] Code follows project conventions
- [ ] Type hints present and correct
- [ ] Docstrings explain complex logic
- [ ] Unit tests have good coverage (80%+)
- [ ] No hardcoded values or magic numbers
- [ ] Error handling is comprehensive
- [ ] Related documentation updated

### Testing Requirements

**Minimum for each task:**
1. Unit tests for new functions/methods
2. Integration tests if cross-module
3. Manual testing checklist completed
4. No regressions in related features

**Target Coverage:**
- New code: 80%+ coverage
- Critical paths: 100% coverage

### Documentation Standards

Update documentation when:
- Adding new features
- Changing existing behavior
- Adding new configuration options
- Modifying API contracts
- Changing deployment procedures

**Files to check:**
- `README.md` - Feature list, installation
- `QUICK_START.md` - Getting started guide
- `CHANGELOG.md` - Version history
- Code docstrings - Function/class documentation

---

## Examples

### Example 1: Small Backend Task

```markdown
# Task 0001: Add Health Check Endpoint

## Metadata

| Field | Value |
|-------|-------|
| **Task Number** | 0001 |
| **Feature** | backend-monitoring |
| **Category** | services |
| **Component(s)** | FastAPI backend |
| **Agent Type** | python-backend-expert |
| **Estimated Size** | S |
| **Status** | pending |
| **Priority** | high |
| **Dependencies** | None |

## Overview

Add a `/health` endpoint to the FastAPI backend to verify it's running and responsive.

## Description

The backend currently has no health check endpoint. This makes it difficult to verify if the backend has started successfully. We need a simple endpoint that returns 200 OK when the backend is healthy.

## Acceptance Criteria

- [ ] GET `/health` endpoint created
- [ ] Returns 200 status code when healthy
- [ ] Returns JSON with status information
- [ ] Response time < 100ms
- [ ] Endpoint documented in API docs

## Implementation Details

### Files to Modify

| File | Type | Purpose |
|------|------|---------|
| `app/main.py` | Modify | Add health endpoint |

### Technical Approach

1. Add `/health` endpoint to FastAPI app
2. Return simple JSON response: `{"status": "ok"}`
3. Keep endpoint lightweight (no database checks)

### Code Example

```python
@app.get("/health")
async def health_check():
    return {"status": "ok"}
```

## Testing Strategy

```python
def test_health_endpoint_returns_200():
    """Test health endpoint returns 200"""
    response = client.get("/health")
    assert response.status_code == 200

def test_health_endpoint_returns_json():
    """Test health endpoint returns JSON"""
    response = client.get("/health")
    assert response.json() == {"status": "ok"}
```

## Implementation Checklist

- [ ] Add endpoint to main.py
- [ ] Write unit tests
- [ ] Run test suite
- [ ] Test manually with curl

## Related Tasks

- Task 0002: Use health check in BackendManager

---
```

### Example 2: Medium UI Task

```markdown
# Task 0004: Implement Dark Mode Toggle

## Metadata

| Field | Value |
|-------|-------|
| **Task Number** | 0004 |
| **Feature** | dark-mode-support |
| **Category** | views |
| **Component(s)** | SettingsView, ThemeManager |
| **Agent Type** | frontend-expert |
| **Estimated Size** | M |
| **Status** | pending |
| **Priority** | medium |
| **Dependencies** | Task 0001, Task 0002, Task 0003 |

## Overview

Add a dark mode toggle to the settings dialog that switches the UI theme between light and dark modes.

## Description

This task implements the UI for dark mode support. The theme infrastructure (created in previous tasks) enables switching. Now we need the toggle control and integration.

## Acceptance Criteria

- [ ] Settings dialog has dark mode toggle switch
- [ ] Toggle persists theme preference
- [ ] All views update immediately when toggled
- [ ] No visual glitches during transition
- [ ] Manual testing checklist completed

## Implementation Details

### Files to Modify

| File | Type | Purpose |
|------|------|---------|
| `views/settings_view.py` | Modify | Add toggle widget |
| `utils/theme_manager.py` | Modify | Implement theme switching |
| `tests/test_views.py` | Modify | Add tests |

### Technical Approach

1. Add QCheckBox to settings dialog
2. Connect toggle signal to ThemeManager.switch_theme()
3. Persist preference in config
4. Load preference on startup

## Testing Strategy

### Manual Testing

- [ ] Toggle appears in settings
- [ ] Clicking toggle changes theme
- [ ] All text is readable in dark mode
- [ ] Theme persists after restart

## Implementation Checklist

- [ ] Add toggle to SettingsView
- [ ] Connect signals
- [ ] Update ThemeManager
- [ ] Test persistence
- [ ] Manual testing

## Related Tasks

- Task 0001: Create ThemeManager
- Task 0002: Implement color schemes
- Task 0003: Create ThemeContext

---
```

### Example 3: Large Feature Task

```markdown
# Task 0010: Implement Batch Video Upload

## Metadata

| Field | Value |
|-------|-------|
| **Task Number** | 0010 |
| **Feature** | batch-upload |
| **Category** | services, controllers, views |
| **Component(s)** | UploadService, TripsController, TripsView |
| **Agent Type** | frontend-expert |
| **Estimated Size** | L |
| **Status** | pending |
| **Priority** | high |
| **Dependencies** | Task 0008, Task 0009 |

## Overview

Enable users to select and upload multiple videos at once, with progress tracking for each file.

## Description

Currently users can only upload one video at a time. This is tedious when processing multiple trip videos. We need batch upload with individual progress tracking.

## Acceptance Criteria

- [ ] Multiple file selection in file dialog
- [ ] Upload queue system
- [ ] Individual progress bars per file
- [ ] Pause/resume capability
- [ ] Error handling per file
- [ ] Overall progress indicator
- [ ] Cancel individual or all uploads
- [ ] Unit and integration tests

## Implementation Details

### Files to Modify

| File | Type | Purpose |
|------|------|---------|
| `services/upload_service.py` | Modify | Add batch upload method |
| `controllers/trips_controller.py` | Modify | Handle upload queue |
| `views/trips_view.py` | Modify | Show progress for multiple files |
| `views/widgets/batch_progress_widget.py` | Create | New widget for batch progress |
| `tests/test_upload_service.py` | Modify | Add batch upload tests |

### Technical Approach

1. Modify file dialog to accept multiple selections
2. Create upload queue in TripsController
3. Process uploads sequentially or in parallel (config)
4. Track progress per file
5. Update UI with batch progress widget

## Testing Strategy

### Unit Tests
- Test queue management
- Test individual file upload
- Test error handling per file
- Test pause/resume logic

### Integration Tests
- Test full batch upload flow
- Test cancellation
- Test error recovery

### Manual Testing
- [ ] Select 5 videos
- [ ] Verify all upload with progress
- [ ] Test pause/resume
- [ ] Test cancel individual
- [ ] Test cancel all
- [ ] Test error handling

## Technical Notes

### Performance Considerations
- Limit concurrent uploads to 3
- Use threading for non-blocking UI
- Optimize for large files (>1GB)

### Potential Challenges
- **Challenge**: Memory usage with large batches
  - Mitigation: Stream uploads, don't load all to memory

- **Challenge**: Network failures mid-batch
  - Mitigation: Implement retry logic per file

## Implementation Checklist

- [ ] Modify file dialog
- [ ] Create batch progress widget
- [ ] Implement queue in controller
- [ ] Add batch upload method
- [ ] Implement pause/resume
- [ ] Add error handling
- [ ] Write tests
- [ ] Manual testing

## Documentation Updates

- [ ] README.md - Add batch upload to features
- [ ] QUICK_START.md - Update upload instructions
- [ ] CHANGELOG.md - Document new feature

---
```

---

## Quick Reference

### Creating a New Feature

1. Create feature directory: `docs/specs/{feature-name}/`
2. Write feature documents:
   - `FEATURE_OVERVIEW.md`
   - `ARCHITECTURE.md`
   - `REQUIREMENTS.md`
   - `TESTING_STRATEGY.md`
3. Create `tasks/` subdirectory
4. Write task files starting from `0001_`

### Creating a New Task

1. Copy `_meta/TASK_TEMPLATE.md`
2. Name it: `{XXXX}_{description}.md`
3. Fill in all sections
4. Save to `{feature-name}/tasks/`
5. Update dependencies in related tasks

### Implementing a Task

1. Run: `/kc:impl {task-number}`
2. System reads task metadata
3. Creates git branch
4. Routes to appropriate agent
5. Agent implements based on acceptance criteria
6. Tests run automatically
7. Code review performed
8. PR created
9. Task status updated

### Task Lifecycle

```
Create Task → pending
     ↓
Start Work → in_progress
     ↓
Complete → completed
```

---

## Support

For questions or issues with the task specification system:

1. Review this guide
2. Check `TASK_TEMPLATE.md` for template
3. See `AGENT_ROLES.md` for agent info
4. Read `WORKFLOW.md` for `/kc:impl` details
5. Refer to example features in `docs/specs/`

---

*Last Updated: 2024*
