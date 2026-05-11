# NanoResearch Skills

This directory contains built-in skills that extend NanoResearch's capabilities.

## Skill Format

Each skill is a directory containing a `SKILL.md` file with:
- YAML frontmatter (name, description, metadata)
- Markdown instructions for the agent

## Attribution

These skills are adapted from [OpenClaw](https://github.com/openclaw/openclaw)'s skill system.
The skill format and metadata structure follow OpenClaw's conventions to maintain compatibility.

Research skills (deep-research, systematic-literature-review, github-deep-research) are adapted from [DeerFlow](https://github.com/bytedance/deer-flow).

## Available Skills

| Skill | Description |
|-------|-------------|
| `deep-research` | Four-phase systematic research methodology with checklist and source prioritization |
| `systematic-literature-review` | Parallel arXiv literature review with APA/IEEE/BibTeX citation formats |
| `github-deep-research` | Multi-round GitHub repository analysis with timeline and metrics |
| `github` | Interact with GitHub using the `gh` CLI |
| `weather` | Get weather info using wttr.in and Open-Meteo |
| `summarize` | Summarize URLs, files, and YouTube videos |
| `tmux` | Remote-control tmux sessions |
| `clawhub` | Search and install skills from ClawHub registry |
| `skill-creator` | Create new skills |
| `rag` | RAG-based knowledge retrieval |
| `memory` | Long-term memory management |
| `cron` | Scheduled task management |