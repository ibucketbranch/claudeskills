---
description: Guide for using the cursor-ai-engineer skill including trigger phrases, common workflows, template selection, and pro tips for effective AI-assisted development.
alwaysApply: false
---

# How to Use This Skill

Practical guide for getting the most out of the cursor-ai-engineer skill.

## Installation

### Option 1: Claude.ai Skills (when available)
```
1. Download cursor-ai-engineer.skill
2. Go to Claude.ai Settings → Skills
3. Upload the .skill file
4. Skill auto-activates on relevant queries
```

### Option 2: Manual Extraction for Cursor IDE
```bash
# The .skill file is a zip archive
unzip cursor-ai-engineer.skill -d ~/skills/

# Copy templates to your project
cp -r ~/skills/cursor-ai-engineer/assets/mdc-templates/*.mdc .cursor/rules/
```

### Option 3: Reference Only
```
Keep the skill files accessible and manually reference them
when working on ML/AI projects in any IDE or context.
```

## Trigger Phrases

The skill activates when you mention:

### Project Setup
- "Set up a new Cursor project"
- "Create Cursor rules for my project"
- "Configure my IDE for ML development"
- "Initialize AI coding rules"

### Code Review
- "Review this code"
- "Check this against Google standards"
- "Code review checklist"
- "Is this code good?"

### ML/AI Development
- "Help me with PyTorch..."
- "Transformers fine-tuning..."
- "Train a model..."
- "CUDA optimization..."
- "MLOps pipeline..."

### Debugging & Optimization
- "Why is my training slow?"
- "Getting OOM errors"
- "Model not learning"
- "Profile my code"

### Learning
- "How do I learn CUDA?"
- "Best courses for ML?"
- "What should I study next?"

## Common Workflows

### Workflow 1: New ML Project Setup

**You say:**
> "I'm starting a new ML project with PyTorch and FastAPI. Set up Cursor rules for me."

**Expected response:**
- Creates `.cursor/rules/` directory
- Copies relevant templates: `core.mdc`, `python.mdc`, `ml-stack.mdc`, `mlops.mdc`
- Customizes based on your stack

**Follow-up prompts:**
- "Add React frontend rules too"
- "I'm also using Supabase"
- "Enable code review rules"

---

### Workflow 2: Code Review

**You say:**
> "Review this training loop against Google standards"

```python
def train(model, data):
    for x, y in data:
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
```

**Expected response:**
- Applies Google code review checklist
- Identifies issues (missing zero_grad, no eval mode, etc.)
- Suggests improvements with code examples

**Follow-up prompts:**
- "What about the complexity?"
- "Are there any pitfalls here?"
- "How would you write this?"

---

### Workflow 3: Debug Training Issues

**You say:**
> "My model training is really slow. What should I check?"

**Expected response:**
- Profiling workflow from expert-knowledge.md
- Common pitfalls checklist
- Specific diagnostic steps

**Follow-up prompts:**
- "How do I use torch profiler?"
- "I think it's memory-bound"
- "Show me DataLoader optimization"

---

### Workflow 4: Architecture Decisions

**You say:**
> "Should I use LoRA or full fine-tuning for my 7B model?"

**Expected response:**
- Decision framework from expert-knowledge.md
- Memory estimates
- Hyperparameter recommendations

**Follow-up prompts:**
- "What about QLoRA?"
- "How much memory will I need?"
- "Show me the LoRA config"

---

### Workflow 5: Learning Path

**You say:**
> "I want to learn CUDA programming. Where should I start?"

**Expected response:**
- Learning roadmap from learning-resources.md
- Recommended courses (NVIDIA DLI, PMPP book)
- Practice resources

**Follow-up prompts:**
- "What about free resources?"
- "How long will it take?"
- "What should I build to practice?"

## Using Templates Directly

### Copy Templates to Project
```bash
# Create Cursor rules directory
mkdir -p .cursor/rules

# Copy all templates (then delete what you don't need)
cp /path/to/skill/assets/mdc-templates/*.mdc .cursor/rules/

# Or copy specific ones
cp /path/to/skill/assets/mdc-templates/core.mdc .cursor/rules/
cp /path/to/skill/assets/mdc-templates/python.mdc .cursor/rules/
cp /path/to/skill/assets/mdc-templates/ml-stack.mdc .cursor/rules/
```

### Customize Templates
Each template has YAML frontmatter you can adjust:

```yaml
---
description: Your custom description here
globs: ["*.py", "src/**/*.py"]  # Adjust file patterns
alwaysApply: true  # or false for on-demand
---
```

### Template Selection Guide

| Project Type | Templates to Use |
|--------------|------------------|
| Python script | core, python |
| FastAPI backend | core, python, sql-supabase |
| React + Python | core, python, react, sql-supabase |
| ML Training | core, python, ml-stack, ml-engineering |
| ML Production | core, python, ml-stack, mlops |
| LLM Fine-tuning | core, python, ml-stack, ai-infra |
| CUDA Development | core, cpp-inference |
| Full ML Platform | ALL templates |

## Asking for Specific References

### Get Expert Knowledge
```
"What are the common PyTorch pitfalls?"
"Show me the debugging workflow for OOM"
"What's the architecture decision guide for distributed training?"
```

### Get Learning Resources
```
"What courses should I take for transformers?"
"Best books for CUDA?"
"Where can I practice ML?"
```

### Get Code Patterns
```
"Show me the PyTorch training loop pattern"
"How do I set up LoRA config?"
"CUDA kernel template for matrix multiply"
```

## Pro Tips

### 1. Be Specific About Your Stack
```
❌ "Help me with ML"
✅ "Help me fine-tune Llama-2-7B with LoRA on a single A100"
```

### 2. Mention Constraints
```
❌ "Train a model"
✅ "Train a model on 16GB GPU with gradient checkpointing"
```

### 3. Ask for Alternatives
```
"What if I only have 8GB VRAM?"
"Is there a simpler approach?"
"What's the trade-off?"
```

### 4. Request Specific Outputs
```
"Give me the exact code"
"Create the .mdc file for this"
"Show me the config file"
```

### 5. Chain Workflows
```
1. "Set up my project" →
2. "Now add training code" →
3. "Review it for issues" →
4. "Optimize for speed"
```

## Quick Command Reference

| Want to... | Say... |
|------------|--------|
| Set up project | "Initialize Cursor rules for [stack]" |
| Review code | "Review this against Google standards" |
| Debug training | "Why is training slow/failing?" |
| Choose architecture | "Should I use X or Y?" |
| Get code pattern | "Show me the pattern for [task]" |
| Learn topic | "How do I learn [topic]?" |
| Find resources | "Best courses/books for [topic]" |
| Optimize | "How do I make this faster?" |
| Estimate memory | "How much memory for [model]?" |

## Combining with Other Tools

### With Web Search
```
"Search for the latest vLLM benchmarks and compare to my setup"
```

### With File Analysis
```
"Read my training script and review it"
[attach file]
```

### With Code Execution
```
"Profile this code and show me the bottlenecks"
```

## Updating the Skill

As frameworks evolve, you may want to update templates:

1. Extract the skill: `unzip cursor-ai-engineer.skill`
2. Edit relevant `.mdc` files
3. Re-package: Use the skill-creator packaging script
4. Re-upload to Claude.ai

Or simply ask:
> "Update the PyTorch template with torch.compile best practices"
