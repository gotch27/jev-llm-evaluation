# Thesis vault agent guide

## About me and how I work

I am Gorazd, soon to graduate from the Faculty of Computer Science and Engineering in Skopje. I use this vault as a living record of my thesis project: where it stands now, how it develops, and what I learn along the way.

I like to work step by step and keep things simple. Help me keep the notes aligned with the actual state of the project, preserving the reasoning behind important decisions and changes. As the project grows, these notes should give me the material and context to form the complete thesis, without having to reconstruct the process from memory.

## Project context

This Obsidian vault supports a graduate thesis exploring Jev, comparing it with LLMs, and potentially combining them. Detailed scope is still being planned. Keep documentation concise and useful for eventual thesis writing.

## Note map

- `Thesis Home.md`: current direction and navigation; start here.
- `Plan.md`: research questions, scope, chapter outline, evaluation approach, milestones, and next actions.
- `Research and Tools.md`: sources, reading notes, models, frameworks, datasets, and reproducible setup.
- `Experiments.md`: experiment questions, methods, configurations, runs, results, limitations, and interpretation.
- `Journal.md`: dated decisions and their reasons, mentor feedback, meaningful changes, and open questions.

## Working habits

- Read the relevant notes and recent journal entries before making changes. Treat the vault as persistent project context.
- After substantive work or an agreed decision, update the relevant note in the same task. Keep proposals and unresolved questions clearly separate from agreed plans.
- Update `Journal.md` when scope or methods change, a decision needs its rationale preserved, mentor feedback arrives, or a finding or blocker changes the next steps. Use dated entries, newest first, with a brief reason and follow-up. Skip routine edits and conversation transcripts.
- Keep the current plan in `Plan.md` and the history of why it changed in `Journal.md`. Link related notes rather than duplicating content.
- Distinguish source claims, hypotheses, and measured findings. Keep source links and access dates; never invent citations, results, or completed work.
- For experiments, record enough to reproduce the run: data and split, model/version, prompts/configuration, code commit, commands, and output locations. Preserve failed and inconclusive results too.
- Keep code, datasets, and raw outputs in the research repository or their actual storage locations; link them here. Never store credentials in the vault.
- Keep the structure flat. Add notes or folders only when existing notes become unwieldy; use Obsidian links and update the home note when needed. Leave `.obsidian` settings alone unless requested.
- End substantive tasks with a brief account of what changed and what remains open. Use specialized skills only when the task calls for them; ordinary Markdown planning and journaling need no extra workflow.
