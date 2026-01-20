---
description: Google engineering practices for code review, CL descriptions, documentation, and handling disagreements. Core principles for professional software development.
alwaysApply: true
---

# Google Engineering Practices Reference

Condensed guidance from Google's public engineering practices, adapted for AI-assisted development.

## Code Review Standard

### The Senior Principle

> A CL should be approved if it improves the overall code health of the system, even if it isn't perfect.

**Key trade-offs:**
- Developers must make progress on tasks
- Reviewers must maintain code health
- No such thing as "perfect" code—only better code
- Seek continuous improvement, not perfection

### What to Look For

**1. Design**
- Does this change belong here?
- Does it integrate well with the system?
- Is now the right time for this change?

**2. Functionality**
- Does it do what the author intended?
- Is behavior good for users (end-users AND future developers)?
- Check edge cases, concurrency issues, UI changes

**3. Complexity**
- Is any line, function, or class too complex?
- "Too complex" = can't be understood quickly by code readers
- Over-engineering: solving problems that don't exist yet

**4. Tests**
- Are tests correct, sensible, and useful?
- Will tests fail when code is broken?
- Will tests produce false positives?
- Each test makes simple, useful assertions

**5. Naming**
- Is the name long enough to communicate purpose?
- Is it short enough to be readable?
- Does it follow project conventions?

**6. Comments**
- Comments explain WHY, not WHAT
- Regular expressions and complex algorithms need explanation
- Update/delete outdated comments with code changes

**7. Style**
- Follow the style guide absolutely
- If not in guide, match existing style
- Personal preference yields to consistency

**8. Documentation**
- Update docs with code changes in same CL
- Delete documentation when deleting code

## Review Speed

### Principles
- One business day maximum for initial response
- Fast responses prevent developer frustration
- Don't interrupt focused work—respond at break points
- Speed does not mean compromising standards

### LGTM with Comments
Approve if:
- Reviewer confident remaining comments are addressable
- Remaining changes are minor
- Author will address all comments

## Writing Good CLs

### Small CLs Are Better
- Reviewed faster and more thoroughly
- Less likely to introduce bugs
- Less wasted work if rejected
- Easier to merge and roll back

### What Belongs in a CL
- One self-contained change
- Everything related to that change (tests, docs, refactoring)
- Not so large that reviewers don't understand it

### CL Description Structure
```
First line: What is being done (imperative)

Body: Why change is being made, context, what approach was taken
and why. Include background links, benchmark results, design doc
references.

Bug: 123456
Test: Describe testing done
```

### Good First Line Examples
- `Delete the FizzBuzz RPC and replace with the new system`
- `Add ability to retrieve users in ViewerServer`
- `Implement GetUserList handler in StatusServer`

### Bad First Line Examples
- `Fix bug` (which bug?)
- `Refactoring` (of what?)
- `Add tests` (for what functionality?)

## Documentation Best Practices

### Documentation Hierarchy
1. **Meaningful names** - Code conveys information without comments
2. **Comments** - Explain decisions, gotchas, "why"
3. **README** - Setup, overview, navigation
4. **Design docs** - Major decisions (archived after implementation)

### When to Document
- Same CL as code change
- Delete docs when code is deleted
- Keep docs close to code they describe

### Writing Principles
- Write for humans first, computers second
- Short sentences, simple words, active voice
- "This is a hammer. You use it to pound nails."

## Handling Disagreements

### Resolution Order
1. Developer and reviewer reach consensus
2. Face-to-face or video meeting
3. Broader team discussion
4. Tech Lead/Manager decision

### Principles
- Technical facts and data overrule opinions
- Style guide is absolute authority on style
- Engineering principles guide design decisions
- Don't let disagreement block progress indefinitely

## Grep Patterns for Quick Lookup

```bash
# Find specific guidance
grep -n "complexity" references/google-engineering.md
grep -n "naming" references/google-engineering.md
grep -n "CL description" references/google-engineering.md
```
