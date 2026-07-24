## Summary

- Describe the change and why it is needed.

## Scope

- [ ] Dependency-free core loop
- [ ] Optional MiniGrid/BabyAI lane
- [ ] Optional PyTorch/GPU lane
- [ ] Fleet scripts or worker protocol
- [ ] Documentation only

## Verification

- [ ] `./scripts/verify.sh`
- [ ] `python3 -m unittest discover -s tests -p 'test_*.py'`
- [ ] `bash -n scripts/*.sh`
- [ ] Optional lane smoke command:

## Reproducibility Notes

- Python runtime pin checked: `.python-version`
- Project dependency manifest checked: `pyproject.toml`
- External dependencies added or changed: yes / no
- Optional dependencies remain outside the default stdlib loop: yes / no

## Research Result

- Hypothesis or issue:
- Config:
- Output summary:
- Decision:

## Risk

- [ ] No bulky `runs/` artifacts committed except curated summaries
- [ ] No hostnames, tokens, SSH material, cookies, or personal data added
- [ ] Remote/GPU work was read-only or explicitly approved
