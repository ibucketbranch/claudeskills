---
name: cursor-ai-engineer
description: Expert AI/ML engineering skill for Cursor IDE grounded in Google engineering practices. Covers the full ML stack from research to production. Use when users want to (1) set up a new Cursor project with AI coding rules, (2) review code using Google's engineering standards, (3) create or customize .mdc/.cursorrules files, (4) work with ML/AI frameworks (PyTorch, Transformers, TensorFlow, scikit-learn), (5) optimize C++ inference code, (6) implement MLOps pipelines, (7) design silicon-aware AI infrastructure, or (8) establish coding standards for Python, React, SQL/Supabase projects. Stack expertise includes C++, Python, NumPy, Pandas, PyTorch (primary), TensorFlow/Keras, Hugging Face Transformers, scikit-learn, FastAPI, CUDA, and production ML systems. Triggers include "set up Cursor," "create AI rules," "review this code," "ML project," "inference optimization," "MLOps," "model serving," or "AI infrastructure."
alwaysApply: true
---

# Cursor AI Engineer

Expert AI/ML engineering skill for Cursor IDE, applying Google's engineering practices across the full ML stack.

## Skill Tiers
```
Baseline:     Python + PyTorch + Transformers
Elite:        C++ + inference optimization  
Production:   MLOps (separates toy projects from real products)
Future:       AI infra + silicon-aware design
```

## Quick Reference

| Task | Template |
|------|----------|
| **How to use this skill** | `references/usage-guide.md` |
| **Expert debugging/profiling** | `references/expert-knowledge.md` |
| **Learning roadmap & courses** | `references/learning-resources.md` |

| Task | Template |
|------|----------|
| New project setup | `core.mdc` + stack-specific templates |
| Python/FastAPI | `python.mdc` |
| React/TypeScript | `react.mdc` |
| SQL/Supabase | `sql-supabase.mdc` |
| ML Stack (PyTorch, TF, sklearn) | `ml-stack.mdc` |
| ML Training/Experiments | `ml-engineering.mdc` |
| C++ Inference Optimization | `cpp-inference.mdc` |
| MLOps/Production | `mlops.mdc` |
| AI Infrastructure | `ai-infra.mdc` |
| Code Review | `code-review.mdc` |

## Core Workflows

### 1. Set Up New Cursor Project

```bash
# Create rules directory
mkdir -p .cursor/rules

# Always include core rules
cp assets/mdc-templates/core.mdc .cursor/rules/

# Add stack-specific rules
cp assets/mdc-templates/python.mdc .cursor/rules/      # Python projects
cp assets/mdc-templates/react.mdc .cursor/rules/       # React/TS projects
cp assets/mdc-templates/sql-supabase.mdc .cursor/rules/ # Database work

# ML/AI projects - add relevant templates
cp assets/mdc-templates/ml-stack.mdc .cursor/rules/     # PyTorch, TF, sklearn
cp assets/mdc-templates/ml-engineering.mdc .cursor/rules/ # Training loops
cp assets/mdc-templates/cpp-inference.mdc .cursor/rules/  # C++ optimization
cp assets/mdc-templates/mlops.mdc .cursor/rules/        # Production ML
cp assets/mdc-templates/ai-infra.mdc .cursor/rules/     # Infrastructure
```

After copying, customize each `.mdc` file:
1. Update project-specific paths and conventions
2. Add team-specific naming patterns
3. Include relevant library versions and APIs

### 2. Code Review (Google Standard)

Apply this checklist to every review:

**Design**
- Does this change belong in this codebase?
- Is it well-integrated with the rest of the system?

**Functionality**
- Does code do what the author intended?
- Is behavior good for users of this code?

**Complexity**
- Could it be simpler?
- Will another developer understand it easily?

**Tests**
- Are there correct, well-designed automated tests?
- Do tests cover edge cases?

**Naming**
- Are names clear and descriptive?
- Do they follow project conventions?

**Comments**
- Are comments clear and explain *why*, not *what*?
- Is documentation updated alongside code?

**Style**
- Does code conform to style guides?
- Is formatting consistent?

For detailed guidance, see `references/google-engineering.md`.

### 3. Create AI Coding Rules

**Rule structure (.mdc format):**
```yaml
---
description: Clear description of when this rule applies
globs: ["*.py", "**/*.py"]  # File patterns
alwaysApply: false  # true = always active, false = agent-requested
---

# Rule Title

Concise instructions for the AI. Use imperative form.
- Specific, actionable guidance
- Examples when helpful
- Constraints and boundaries
```

**Best practices:**
- Keep rules under 500 lines
- One concern per rule file
- Use globs to auto-attach rules to relevant files
- Write for another Claude/AI instance, not humans

### 4. Effective Cursor Prompting

**Prompt structure:**
```
[Context]: What exists, what's the goal
[Task]: Specific action to take
[Constraints]: Boundaries, patterns to follow
[Verification]: How to confirm success
```

**Example:**
```
Context: FastAPI backend with SQLAlchemy, auth via Supabase
Task: Add endpoint for user preferences CRUD
Constraints: Follow existing patterns in routes/users.py, use Pydantic models
Verification: Write tests first, then implement until tests pass
```

**Mode selection:**
- **Agent Mode** (⌘.): Multi-file changes, refactoring, new features
- **Chat Mode**: Questions, explanations, design discussions
- **Inline Edit** (⌘K): Single-location focused edits

For detailed workflows, see `references/cursor-workflows.md`.

## File Organization

```
.cursor/
├── rules/
│   ├── core.mdc           # Always-on foundational rules
│   ├── python.mdc         # Python-specific (auto-attach *.py)
│   ├── react.mdc          # React-specific (auto-attach *.tsx)
│   ├── sql.mdc            # SQL-specific
│   ├── ml.mdc             # ML engineering rules
│   └── code-review.mdc    # Review workflow rules
└── .cursorignore          # Files to exclude from AI context
```

## Key Principles (Google Engineering)

1. **Continuous improvement over perfection** - Approve CLs that improve overall code health
2. **Technical facts over opinions** - Base decisions on data and engineering principles
3. **Style guide is authority** - On style matters, defer to the guide
4. **Implement what's needed now** - Avoid speculative features
5. **Tests and docs in same CL** - Keep changes atomic and complete

## Template Assets

Located in `assets/mdc-templates/`:

### Foundation
- `core.mdc` - Always-on foundational rules (context-first, naming, testing)

### Languages & Frameworks
- `python.mdc` - Python/FastAPI/Pydantic patterns
- `react.mdc` - React/TypeScript/Tailwind patterns
- `sql-supabase.mdc` - SQL, Supabase, MySQL patterns
- `cpp-inference.mdc` - C++ inference optimization, CUDA, SIMD

### ML/AI Stack
- `ml-stack.mdc` - NumPy, Pandas, PyTorch, TensorFlow, Transformers, scikit-learn
- `ml-engineering.mdc` - Training loops, experiment tracking, model development

### Production & Infrastructure
- `mlops.mdc` - DVC, MLflow, CI/CD, feature stores, monitoring
- `ai-infra.mdc` - Silicon-aware design, GPU optimization, model serving

### Process
- `code-review.mdc` - Google-style review workflow

## Framework Priority (ML/AI)

1. **PyTorch** — Primary framework (🔥 most important)
2. **Hugging Face Transformers** — NLP/LLM tasks
3. **TensorFlow/Keras** — When required by existing code
4. **Scikit-learn** — Classical ML, preprocessing, evaluation

## Skill Progression Path

```
1. Baseline: Python + PyTorch + Transformers
   → Core ML development, model training, fine-tuning

2. Elite: + C++ inference optimization
   → SIMD, CUDA kernels, quantization, low-latency serving

3. Production: + MLOps
   → DVC, MLflow, CI/CD, monitoring, feature stores

4. Future: + AI infrastructure
   → Multi-GPU, TensorRT, vLLM, silicon-aware design
```

Copy and customize these templates for each project.

## Expert-Level Guidance

For advanced troubleshooting and deep knowledge, see `references/expert-knowledge.md`:

**Debugging Workflows:**
- NaN/Inf in training
- Out of Memory (OOM)
- Slow training diagnosis
- Model not learning

**Performance Profiling:**
- PyTorch profiler usage
- NVIDIA Nsight Systems
- Roofline analysis
- Bottleneck identification

**Common Pitfalls:**
- PyTorch gotchas (no_grad, data transfer, vectorization)
- CUDA pitfalls (coalescing, bank conflicts, warp divergence)
- Training bugs (gradient accumulation, data leakage)

**Architecture Decisions:**
- Framework selection guide
- Training strategy selection (DDP vs FSDP vs DeepSpeed)
- PEFT method selection (LoRA vs QLoRA vs adapters)
- Quantization decision tree

**Memory Estimation:**
- Model memory formulas
- Activation memory calculation
- Quick estimates for common model sizes

**Hyperparameter Ranges:**
- Learning rates by optimizer and task
- Batch sizes by domain
- LoRA hyperparameters (r, alpha, dropout)

## Learning & Deep Dive Resources

For continued learning, see `references/learning-resources.md`:

**Top Free Courses:**
- fast.ai - Practical Deep Learning
- Hugging Face NLP/LLM Courses
- Andrej Karpathy - Neural Networks: Zero to Hero
- Full Stack Deep Learning

**Essential Books:**
- "Deep Learning" (Goodfellow) - free at deeplearningbook.org
- "Dive into Deep Learning" - free at d2l.ai
- "Programming Massively Parallel Processors" (CUDA bible)
- "Designing Machine Learning Systems" (Huyen)

**Key Papers:**
- Attention Is All You Need (Transformers)
- LoRA / QLoRA (efficient fine-tuning)
- FlashAttention (memory-efficient attention)

**Communities:**
- Hugging Face Discord
- PyTorch Forums
- r/MachineLearning, r/LocalLLaMA

**Practice Platforms:**
- Kaggle (competitions + free GPUs)
- Google Colab (free T4)

**Learning Roadmap:**
Foundations → PyTorch → Transformers → MLOps → Specialization
