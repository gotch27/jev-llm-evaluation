# Thesis vault agent guide

## About me and how I work

I am Gorazd. I use this vault as a living record of my thesis project: where it stands now, how it develops, and what I learn along the way.

I like to work step by step and keep things simple. Help me keep the notes aligned with the actual state of the project, preserving the reasoning behind important decisions and changes. As the project grows, these notes should give me the material and context to form the complete thesis, without having to reconstruct the process from memory.

## Project context

This Obsidian vault holds the context of a graduate thesis exploring Jev and its capabilities through evaluation, comparison with large language models (LLMs), and experiments. Keep documentation concise and useful for thesis writing.

## Note map

- `Thesis Home.md`: current direction and navigation; start here.
- `Plan.md`: research questions, scope, chapter outline, evaluation approach, milestones, next actions, and current open questions approved for recording.
- `Research and Tools.md`: sources, reading notes, models, frameworks, datasets, and reproducible setup.
- `Use Cases.md`: overview linking to individual use cases, shared evaluation context, and experiment-recording guidelines.
- `Use Cases/`: notes documenting published evaluations and separate notes for our own use cases. Published-evaluation notes contain descriptions, methods, sources, and limitations; our use-case notes contain descriptions, experiments, results, and interpretation. Keep these separate even when the tasks are similar.
- `Journal.md`: dated decisions and their reasons, mentor feedback, and meaningful changes.

## How we collaborate

- Treat the vault as Gorazd's reviewed and authorized project record. Do not introduce inferred decisions, speculative plans, or assumptions about project status. Keep standing instructions focused on lasting purpose and working rules; date changing status in the relevant project note.
- Read the relevant notes and recent journal entries before project research, discussion, or editing. Reading context does not require permission to save changes.
- Chat first. Keep research, brainstorming, recommendations, and drafts in chat until Gorazd explicitly authorizes saving them to the vault.
- Edit vault files only when explicitly requested or permitted, including this `AGENTS.md`. Agreement with an idea is not permission to edit files. Requests to explore, research, discuss, or plan do not authorize vault changes. Marking a draft as provisional does not change this rule.
- Work through one decision at a time. Do not expand an exploratory discussion into a complete plan before we have agreed on the direction.
- Explain options and trade-offs, and distinguish recommendations from agreed decisions. Gorazd chooses the direction.
- At natural stopping points, briefly recap what we have agreed and what remains open in chat, so Gorazd can decide what deserves saving.
- When saving is authorized, keep changes within the authorized scope and save the agreed material. Do not automatically update other notes or the journal. Briefly report which files changed.
- Permission to edit files does not authorize Git commits or pushes. Commit or push only when explicitly requested or permitted.

## Working with notes when saving is authorized

- Save proposals, hypotheses, and unresolved questions only when Gorazd explicitly approves recording them, and label them accurately. Approval to record a claim does not make it a verified finding. Never replace genuine research uncertainty with an unsupported assertion.
- When a journal update is authorized, record scope or method changes, decision rationale, mentor feedback, or findings and blockers that change the next steps. Use dated entries, newest first, with a brief reason and follow-up. Skip routine edits and conversation transcripts.
- Keep the current plan and approved open questions in `Plan.md`, and dated history and rationale in `Journal.md`. Link related notes rather than maintaining duplicate lists.
- Distinguish source claims, hypotheses, and measured findings. Keep source links and access dates; never invent citations, results, or completed work.
- For experiments, record enough to reproduce the run: data and split, model/version, prompts/configuration, code commit, commands, and output locations. Preserve failed and inconclusive results too.
- Keep code, datasets, and raw outputs in the research repository or their actual storage locations; link them here. Never store credentials in the vault.
- Keep the main project notes at the root and use-case notes under `Use Cases/`, with each of our cases’ experiments, results, and interpretation together, separate from notes documenting published evaluations. Propose further splits only when existing notes become unwieldy; create them and update navigation links only within an authorized change. Leave `.obsidian` settings alone unless requested.
- End substantive tasks with a brief account of what changed and what remains open. Use specialized skills only when the task calls for them; ordinary Markdown planning and journaling need no extra workflow.
