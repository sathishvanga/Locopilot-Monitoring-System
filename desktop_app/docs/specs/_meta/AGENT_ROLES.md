# Agent Roles and Responsibilities

## Overview

The `/kc:impl` workflow uses specialized agents to implement tasks based on their type. Each agent has specific expertise, responsibilities, and quality standards.

This document defines the three primary agent types used in the Locopilot Desktop Application task system.

---

## Agent: python-backend-expert

### Description

Specialized agent for Python backend development, including services, models, business logic, and backend infrastructure.

### Suitable For

- **Service Layer**: API integration, backend business logic, external service wrappers
- **Data Models**: Pydantic models, validation logic, data structures
- **Utilities**: Helper functions, configuration management, logging setup
- **Backend Processing**: Video processing, ML model integration, data transformation
- **API Endpoints**: FastAPI route handlers, request/response models

### Typical Files Modified

```
services/
├── auth_service.py
├── trip_service.py
├── upload_service.py
├── local_processing_service.py
└── backend_manager.py

models/
├── auth_models.py
└── trip_models.py

utils/
├── config.py
├── logger.py
└── api_client.py

tests/
├── test_services.py
└── test_models.py
```

### Responsibilities

1. **Implementation**
   - Write clean, well-structured Python code
   - Follow PEP 8 style guidelines
   - Use type hints for all functions and methods
   - Implement proper error handling
   - Follow existing service patterns

2. **Testing**
   - Write comprehensive unit tests
   - Achieve 80%+ code coverage
   - Use pytest or unittest framework
   - Mock external dependencies appropriately
   - Test edge cases and error conditions

3. **Documentation**
   - Add docstrings to all classes and functions
   - Document complex logic with inline comments
   - Update API documentation if endpoints change
   - Keep code self-documenting when possible

4. **Quality Standards**
   - No hardcoded values (use config)
   - Proper exception handling
   - Logging at appropriate levels
   - Thread-safe code where needed
   - Efficient algorithms and data structures

### Code Patterns to Follow

#### Service Pattern

```python
class ServiceName:
    """
    Service for [purpose]

    Handles [responsibilities]
    """

    def __init__(self):
        """Initialize service"""
        self.logger = get_logger(__name__)
        self.settings = get_settings()

    def method_name(self, param: type) -> tuple[bool, Optional[DataType], Optional[str]]:
        """
        [Method description]

        Args:
            param: [Description]

        Returns:
            tuple: (success: bool, data: Optional[DataType], error: Optional[str])
        """
        try:
            # Implementation
            return True, data, None
        except Exception as e:
            self.logger.error(f"Error: {e}", exc_info=True)
            return False, None, str(e)
```

#### Model Pattern

```python
from pydantic import BaseModel, Field

class ModelName(BaseModel):
    """
    [Model description]
    """
    field_name: str = Field(..., description="Field description")
    optional_field: Optional[int] = Field(None, description="Optional field")

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "field_name": "example value",
                "optional_field": 42
            }
        }
```

### Examples of Tasks

**Small (S):**
- Add new endpoint to existing service
- Create new Pydantic model
- Add logging to existing function
- Fix bug in service method

**Medium (M):**
- Implement new service class
- Add retry logic to API calls
- Create integration with new API
- Refactor service for better structure

**Large (L):**
- Build complete feature service
- Implement complex business logic
- Major refactoring of backend layer
- Add comprehensive error recovery

---

## Agent: frontend-expert

### Description

Specialized agent for PySide6 frontend development, including views, widgets, controllers, and UI/UX implementation.

### Suitable For

- **Views**: Main application windows, dialog boxes, forms
- **Widgets**: Custom UI components, progress indicators, interactive elements
- **Controllers**: UI business logic, signal/slot connections, worker threads
- **User Experience**: Layout design, responsive UI, accessibility
- **Visual Elements**: Styling, themes, icons, resources

### Typical Files Modified

```
views/
├── login_view.py
├── trips_view.py
└── widgets/
    └── progress_widget.py

controllers/
├── login_controller.py
└── trips_controller.py

resources/
└── logos/

tests/
└── test_views.py
```

### Responsibilities

1. **Implementation**
   - Build responsive, user-friendly interfaces
   - Follow Qt best practices
   - Implement proper signal/slot patterns
   - Use worker threads for long operations
   - Create reusable widgets

2. **User Experience**
   - Ensure intuitive workflows
   - Provide clear feedback to users
   - Handle errors gracefully with dialogs
   - Maintain consistent UI patterns
   - Support accessibility features

3. **Testing**
   - Write unit tests for controllers
   - Test signal/slot connections
   - Verify UI updates correctly
   - Test edge cases and error states
   - Manual testing of UI flows

4. **Quality Standards**
   - No UI blocking operations
   - Proper resource cleanup
   - Memory leak prevention
   - Consistent styling
   - Cross-platform compatibility

### Code Patterns to Follow

#### Controller Pattern

```python
from PySide6.QtCore import QObject, Signal, QThread

class ControllerName(QObject):
    """
    Controller for [purpose]

    Handles [responsibilities]
    """

    # Signals
    operation_success = Signal(object)
    operation_error = Signal(str)

    def __init__(self, service, view):
        """Initialize controller"""
        super().__init__()
        self.service = service
        self.view = view
        self.logger = get_logger(__name__)

        # Connect signals
        self._connect_signals()

    def _connect_signals(self):
        """Connect view signals to controller slots"""
        self.view.button_clicked.connect(self.handle_button_click)

    def handle_button_click(self):
        """Handle button click event"""
        # Start worker thread for long operation
        worker = OperationWorker(self.service)
        worker.finished.connect(self.on_operation_complete)
        worker.start()
```

#### Worker Thread Pattern

```python
class OperationWorker(QThread):
    """
    Worker thread for [operation]
    """

    finished = Signal(bool, object, str)  # success, data, error

    def __init__(self, service):
        super().__init__()
        self.service = service

    def run(self):
        """Execute operation in background thread"""
        try:
            success, data, error = self.service.perform_operation()
            self.finished.emit(success, data, error or "")
        except Exception as e:
            self.finished.emit(False, None, str(e))
```

### Examples of Tasks

**Small (S):**
- Add button to existing view
- Update label text
- Add icon to widget
- Fix layout spacing

**Medium (M):**
- Create new dialog window
- Implement custom widget
- Add progress tracking UI
- Refactor view for better UX

**Large (L):**
- Build complete feature UI
- Implement complex workflow
- Major UI redesign
- Add comprehensive theming

---

## Agent: code-reviewer

### Description

Specialized agent for code review, quality assurance, testing verification, and best practices enforcement.

### Suitable For

- **Code Review**: Reviewing implemented code against standards
- **Quality Checks**: Verifying code quality and conventions
- **Test Verification**: Ensuring adequate test coverage
- **Documentation Review**: Checking documentation completeness
- **Best Practices**: Enforcing project patterns and standards

### Responsibilities

1. **Code Quality Review**
   - Check code follows project conventions
   - Verify proper error handling
   - Ensure type hints are present
   - Review for security vulnerabilities
   - Check for code smells and anti-patterns

2. **Test Coverage Review**
   - Verify unit tests exist for new code
   - Check test coverage percentage
   - Review test quality and completeness
   - Ensure edge cases are tested
   - Validate integration tests if needed

3. **Documentation Review**
   - Verify docstrings are present
   - Check documentation is accurate
   - Ensure README is updated if needed
   - Review CHANGELOG entries
   - Validate code comments

4. **Standards Enforcement**
   - Ensure acceptance criteria are met
   - Verify no regressions introduced
   - Check dependencies are appropriate
   - Review performance implications
   - Validate architectural decisions

### Review Checklist

#### Code Quality
- [ ] Code follows PEP 8 (Python) or project style guide
- [ ] All functions have type hints
- [ ] Complex logic has explanatory comments
- [ ] No hardcoded values (use config)
- [ ] Error handling is comprehensive
- [ ] Logging is appropriate
- [ ] No security vulnerabilities

#### Testing
- [ ] Unit tests cover new functionality
- [ ] Test coverage is 80%+ for new code
- [ ] Edge cases are tested
- [ ] Error conditions are tested
- [ ] Integration tests added if needed
- [ ] All tests pass

#### Documentation
- [ ] Docstrings added to new functions/classes
- [ ] README updated if feature visible to users
- [ ] CHANGELOG has entry for changes
- [ ] Code is self-documenting where possible
- [ ] Complex algorithms explained

#### Architecture
- [ ] Follows existing patterns
- [ ] No tight coupling introduced
- [ ] Dependencies are appropriate
- [ ] Performance is acceptable
- [ ] Thread-safety considered

#### Acceptance Criteria
- [ ] All acceptance criteria from task are met
- [ ] No regressions in related features
- [ ] Manual testing completed
- [ ] Edge cases handled

### Review Outcomes

**Approve**
- All criteria met
- Code quality is high
- Tests are comprehensive
- Documentation is complete

**Request Changes**
- Missing tests
- Documentation incomplete
- Code quality issues
- Acceptance criteria not met
- Security concerns

**Comment**
- Suggestions for improvement
- Alternative approaches
- Performance considerations
- Future refactoring ideas

### Examples of Review Comments

**Code Quality:**
```
The error handling here should catch more specific exceptions.
Consider catching ConnectionError and TimeoutError separately
to provide better error messages to users.
```

**Testing:**
```
Please add tests for the edge case where the video file is empty.
This could cause issues in production.
```

**Documentation:**
```
The process_video() function would benefit from a docstring
explaining the return value format and possible exceptions.
```

**Performance:**
```
Loading all trips into memory could be problematic with large datasets.
Consider implementing pagination or lazy loading.
```

---

## Agent Selection Guidelines

### When to Use python-backend-expert

Use for tasks that primarily involve:
- Service layer changes
- Backend models
- API integration
- Utility functions
- Backend processing logic

**Example Task Titles:**
- "Implement trip status update service"
- "Add health check endpoint"
- "Create video processing service"
- "Fix backend startup failure"

### When to Use frontend-expert

Use for tasks that primarily involve:
- UI components
- Views and dialogs
- Controllers
- User interactions
- Visual design

**Example Task Titles:**
- "Implement dark mode toggle"
- "Create batch upload progress widget"
- "Add error message dialog"
- "Refactor trips view layout"

### When to Use code-reviewer

The code-reviewer agent is automatically invoked after implementation by either python-backend-expert or frontend-expert agents. It reviews the implemented code and provides feedback.

**Automatic Usage:**
- After task implementation completes
- Before PR creation
- As part of `/kc:impl` workflow

---

## Multi-Agent Tasks

Some tasks may require multiple agents working in sequence:

### Example: Full-Stack Feature

```
1. python-backend-expert: Implement backend service (Task 0001)
2. frontend-expert: Create UI for feature (Task 0002)
3. code-reviewer: Review both implementations
```

### Example: Complex Feature

```
1. python-backend-expert: Create data models (Task 0001)
2. python-backend-expert: Implement service logic (Task 0002)
3. frontend-expert: Build UI components (Task 0003)
4. frontend-expert: Integrate with backend (Task 0004)
5. code-reviewer: Comprehensive review
```

---

## Quality Standards Summary

### All Agents Must

- [ ] Follow project coding conventions
- [ ] Write comprehensive tests
- [ ] Add appropriate documentation
- [ ] Ensure no regressions
- [ ] Meet all acceptance criteria
- [ ] Provide clear commit messages
- [ ] Update relevant documentation files

### Python Code Standards

- [ ] PEP 8 compliant
- [ ] Type hints on all functions
- [ ] Docstrings on classes and methods
- [ ] Proper exception handling
- [ ] No hardcoded values
- [ ] Appropriate logging

### UI Code Standards

- [ ] Non-blocking operations
- [ ] Proper signal/slot usage
- [ ] Worker threads for long tasks
- [ ] Clear user feedback
- [ ] Error handling with dialogs
- [ ] Consistent styling

### Test Standards

- [ ] 80%+ coverage for new code
- [ ] Unit tests for all functions
- [ ] Integration tests where needed
- [ ] Edge cases covered
- [ ] Error conditions tested
- [ ] Mock external dependencies

---

## Related Documentation

- See `SPECIFICATION_GUIDE.md` for task creation guidelines
- See `TASK_TEMPLATE.md` for task file structure
- See `WORKFLOW.md` for `/kc:impl` execution details

---

*Last Updated: 2024*
