# Codex Agents

These files are generated from the repo's Claude agent prompts under `.claude/agents/`.

## Regenerate

```bash
python3 scripts/convert_claude_agents_to_codex.py
```

## Install Into Codex

```bash
./scripts/install_codex_agents.sh
```

The install script copies `.codex/agents/*.toml` into `~/.codex/agents` and backs up any overwritten files to `~/.codex/agent-backups/<timestamp>/`.
