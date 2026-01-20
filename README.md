# 🧠 Claude Skills & Cursor Rules

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Cursor Compatible](https://img.shields.io/badge/Cursor-Compatible-purple.svg)](https://cursor.sh)
[![Claude Skills](https://img.shields.io/badge/Claude-Skills-orange.svg)](https://claude.ai)

> **Production-grade AI coding rules grounded in Google engineering practices.**  
> A curated collection of Claude Skills and Cursor IDE rules for ML/AI engineering, code quality, and developer productivity.

---

## 🎯 What This Is

This repository contains **battle-tested rules and knowledge bases** that supercharge AI-assisted development:

- **Claude Skills** — Portable AI expertise for Claude.ai conversations
- **Cursor Rules** (`.mdc` files) — Auto-attach coding standards for Cursor IDE
- **Reference Docs** — Deep knowledge on debugging, architecture decisions, and learning paths

Whether you're training models, building APIs, or reviewing code, these rules ensure consistent, high-quality AI assistance.

---

## 📦 What's Included

### 🔧 Core Rules (Always Active)

| File | Purpose |
|------|---------|
| `core.mdc` | Foundational coding standards — context-first development, naming, testing |
| `code-review.mdc` | Google-style code review checklist |
| `SKILL.md` | Master skill definition — ML/AI engineering expertise |
| `google-engineering.md` | Google engineering principles condensed |

### 🎨 Language & Framework Rules (Auto-Attach)

| File | Triggers On | Covers |
|------|-------------|--------|
| `python.mdc` | `*.py` | PEP 8, type hints, FastAPI, Pydantic, pytest |
| `react.mdc` | `*.tsx`, `*.jsx` | Hooks, TypeScript, Tailwind, testing |
| `sql-supabase.mdc` | `*.sql` | PostgreSQL, Supabase, RLS, migrations |
| `cpp-inference.mdc` | `*.cpp`, `*.cu` | SIMD, CUDA kernels, quantization |

### 🤖 ML/AI Stack Rules

| File | Purpose |
|------|---------|
| `ml-stack.mdc` | PyTorch, TensorFlow, Transformers, scikit-learn patterns |
| `ml-engineering.mdc` | Training loops, experiment tracking, reproducibility |
| `mlops.mdc` | DVC, MLflow, CI/CD, monitoring, feature stores |
| `ai-infra.mdc` | GPU optimization, model serving, silicon-aware design |

### 📚 Reference Knowledge

| File | Contains |
|------|----------|
| `expert-knowledge.md` | Debugging workflows, profiling, common pitfalls, memory estimation |
| `learning-resources.md` | Curated courses, books, papers, communities |
| `cursor-workflows.md` | Effective prompting, mode selection, context management |
| `usage-guide.md` | How to use this skill collection |

---

## 🚀 Quick Start

### For Cursor IDE

```bash
# Clone to your home directory
git clone https://github.com/ibucketbranch/claudeskills.git ~/.cursor-rules

# Symlink to Cursor's global rules location
ln -sf ~/.cursor-rules ~/.cursor/rules

# Or copy to a specific project
cp -r ~/.cursor-rules/*.mdc /path/to/project/.cursor/rules/
```

### For Claude.ai

1. Copy the contents of any `.md` or `.mdc` file
2. Add as a **User Rule** in Claude settings, or
3. Paste directly into conversation context

### Per-Project Setup

```bash
# In your project root
mkdir -p .cursor/rules

# Copy what you need
cp ~/.cursor-rules/core.mdc .cursor/rules/
cp ~/.cursor-rules/python.mdc .cursor/rules/
cp ~/.cursor-rules/ml-stack.mdc .cursor/rules/
```

---

## 💡 How It Works

### Rule Types

| Type | `alwaysApply` | Behavior |
|------|---------------|----------|
| **Always-On** | `true` | Injected into every prompt |
| **Auto-Attach** | `false` + `globs` | Activates when editing matching files |
| **On-Demand** | `false` | AI pulls in when contextually relevant |

### Example Rule Structure

```yaml
---
description: Python development standards
globs: ["*.py", "**/*.py"]
alwaysApply: false
---

# Python Engineering Rules

## Code Style
- Follow PEP 8 with 88-char line length (Black compatible)
- Use type hints for all function signatures
- Prefer f-strings over .format()
```

---

## 🎓 Skill Progression Path

```
┌─────────────────────────────────────────────────────────────┐
│  1. BASELINE        Python + PyTorch + Transformers        │
│                     Core ML development, fine-tuning        │
├─────────────────────────────────────────────────────────────┤
│  2. ELITE           + C++ inference optimization            │
│                     SIMD, CUDA, quantization, low-latency   │
├─────────────────────────────────────────────────────────────┤
│  3. PRODUCTION      + MLOps                                 │
│                     DVC, MLflow, CI/CD, monitoring          │
├─────────────────────────────────────────────────────────────┤
│  4. INFRASTRUCTURE  + AI infra                              │
│                     Multi-GPU, TensorRT, vLLM, silicon-aware│
└─────────────────────────────────────────────────────────────┘
```

---

## 🗣️ Example Prompts

Once rules are active, try these:

| Goal | Prompt |
|------|--------|
| **Code Review** | "Review this code against Google standards" |
| **Debug Training** | "My model training is slow. What should I check?" |
| **Architecture** | "Should I use LoRA or full fine-tuning for 7B?" |
| **Memory Estimate** | "How much VRAM for Llama-2-13B inference?" |
| **Learn Topic** | "Best resources to learn CUDA programming?" |

---

## 🛠️ Customization

### Adjust Auto-Attach Patterns

Edit the `globs` field to match your project structure:

```yaml
---
globs: ["src/**/*.py", "lib/**/*.py"]  # Only src and lib folders
---
```

### Make a Rule Always-On

```yaml
---
alwaysApply: true  # Now active in every conversation
---
```

### Add Project-Specific Conventions

Extend any rule with your team's patterns:

```markdown
## Project-Specific

- Use `src/` for all source code
- Database models in `src/models/`
- API routes follow `/api/v1/` prefix
```

---

## 📊 Template Selection Guide

| Project Type | Recommended Templates |
|--------------|----------------------|
| Python CLI/Script | `core`, `python` |
| FastAPI Backend | `core`, `python`, `sql-supabase` |
| React + Python | `core`, `python`, `react`, `sql-supabase` |
| ML Training | `core`, `python`, `ml-stack`, `ml-engineering` |
| ML Production | `core`, `python`, `ml-stack`, `mlops` |
| LLM Fine-tuning | `core`, `python`, `ml-stack`, `ai-infra` |
| CUDA Development | `core`, `cpp-inference` |
| Full ML Platform | ALL templates |

---

## 🔗 Key Principles

These rules encode best practices from:

1. **Google Engineering Practices** — Code review, CL descriptions, style guides
2. **PyTorch Best Practices** — Training loops, memory optimization, debugging
3. **Production ML** — MLOps, monitoring, reproducibility
4. **Modern Development** — Type safety, testing, documentation

---

## 📝 License

MIT License — use freely, modify as needed, attribution appreciated.

---

## 🤝 Contributing

Found an improvement? PRs welcome:

1. Fork the repo
2. Create a feature branch
3. Submit a PR with clear description

---

## 👤 Author

**Michael Valderrama**  
ML/AI Engineer focused on production systems and developer tooling.

- GitHub: [@ibucketbranch](https://github.com/ibucketbranch)

---

<p align="center">
  <i>Making AI-assisted development more consistent, one rule at a time.</i>
</p>
