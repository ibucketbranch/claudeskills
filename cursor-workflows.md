---
description: Cursor IDE workflow patterns including Agent Mode, Chat Mode, Inline Edit, rules system, context management, effective prompting, and test-driven AI development.
alwaysApply: false
---

# Cursor IDE Workflows Reference

Effective patterns for AI-assisted development in Cursor.

## Cursor Modes

### Agent Mode (⌘.)
**Use for:** Multi-file changes, refactoring, new features, complex tasks

**Capabilities:**
- Auto-pulls context from codebase
- Executes terminal commands
- Creates/edits multiple files
- Runs tests and iterates

**Best prompt pattern:**
```
Context: [Describe current state]
Task: [Specific goal]
Constraints: [Patterns to follow, files to reference]
Verification: Write tests first, implement until tests pass
```

### Chat Mode
**Use for:** Questions, explanations, design discussions, learning

**When to use:**
- Understanding existing code
- Exploring approaches before implementing
- Debugging conceptual issues
- Getting explanations without changes

### Inline Edit (⌘K)
**Use for:** Single-location focused edits

**Best for:**
- Quick function modifications
- Adding a specific method
- Fixing a single bug
- Refactoring one section

## Rules System

### Rule Types

**Project Rules** (`.cursor/rules/*.mdc`)
- Version-controlled, team-shared
- Can auto-attach via glob patterns
- Loaded based on file context

**User Rules** (Cursor Settings > Rules)
- Global to your environment
- Personal preferences
- Not project-specific

### Rule Priority Order
1. Local (manually included with @ruleName)
2. Auto-attached (matching glob patterns)
3. Agent-requested (AI determines relevance)
4. User Rules (always applied last)

### .mdc File Structure
```yaml
---
description: When AI should use this rule
globs: ["*.py", "**/*.py"]  # Optional: auto-attach to files
alwaysApply: false          # true = always on, false = agent-requested
---

# Rule Content

Instructions in markdown format.
Use imperative voice.
Be specific and actionable.
```

## Context Management

### @ References
- `@filename.ts` - Include specific file
- `@folder/` - Include folder contents
- `@web` - Enable web search
- `@docs` - Search documentation
- `@codebase` - Search entire codebase

### .cursorignore
Exclude files from AI context:
```
# Dependencies
node_modules/
venv/
__pycache__/

# Build artifacts
dist/
build/
*.pyc

# Large/binary files
*.pdf
*.zip
data/

# Sensitive
.env
secrets/
```

## Effective Prompting

### Test-Driven AI Development
Most reliable pattern for non-trivial features:
```
Write tests first, then the code, then run the tests
and update the code until tests pass.
```

This eliminates the QA tester loop and forces verification.

### Context-First Prompting
```
1. UNDERSTAND BEFORE CODING
   - List files in target directory
   - Identify existing patterns
   - Detect environment variables and configs

2. MATCH EXISTING STYLE
   - Follow naming conventions found in codebase
   - Use same libraries/patterns already present
   - Maintain consistent structure

3. VERIFY BEFORE COMPLETING
   - Run tests
   - Check for type errors
   - Confirm integration works
```

### Prompt Templates

**New Feature:**
```
Context: [Tech stack, existing patterns]
Feature: [What to build]
Location: Follow patterns in @path/to/similar.ts
Requirements:
- [Requirement 1]
- [Requirement 2]
Tests: Write tests first, implement until passing
```

**Bug Fix:**
```
Bug: [Description of issue]
Expected: [What should happen]
Actual: [What happens instead]
Reproduce: [Steps or test case]
Fix: Investigate @relevant/files, propose minimal fix
```

**Refactor:**
```
Target: @path/to/file.py
Goal: [What improvement]
Constraints:
- Maintain all existing functionality
- Keep same public API
- Add tests for any untested code paths
```

## Workflow Patterns

### Feature Development
1. Create/update design doc (if complex)
2. Write or update `.mdc` rules if needed
3. Prompt Agent Mode with test-first approach
4. Review generated code against checklist
5. Iterate with specific feedback
6. Final review before commit

### Code Review Integration
1. Open PR/CL changes in Cursor
2. Use Chat mode to explain changes
3. Apply review checklist (design, functionality, complexity, tests, naming, comments, style)
4. Request fixes via Agent mode with specific feedback
5. Verify fixes before approving

### Debugging Workflow
1. Reproduce issue with test case if possible
2. Use `@terminal` context to show errors
3. Prompt: "Investigate this error, check @relevant/files"
4. Request minimal fix, not rewrites
5. Verify fix doesn't break other tests

## Model Selection

**For complex reasoning:**
- Claude Sonnet 4 / Opus 4.5
- OpenAI o4-mini

**For fast iteration:**
- Claude Haiku
- GPT-4o-mini

**Rule of thumb:**
- Start with faster model
- Escalate to stronger model if stuck
- Complex architecture = stronger model

## Common Pitfalls

### Avoid
- Vague prompts ("make it better")
- Accepting first output without review
- Letting AI rewrite entire files unnecessarily
- Ignoring test failures
- Not reading generated code

### Do
- Be specific about constraints
- Review like a teammate's PR
- Iterate with targeted feedback
- Keep changes small and focused
- Verify functionality manually
