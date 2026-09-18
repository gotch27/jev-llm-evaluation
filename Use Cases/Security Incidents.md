# Security Incidents

[[Use Cases|← Use cases]]

Added on 2026-09-18 to document TypeSafe’s published evaluation as background for the thesis. Any independent experiments on a similar task will be documented in a separate note.

## TypeSafe's published evaluation

**Task:** Decide how to respond to a security alert.

**Inputs:** The alert, asset details, open tickets, registered devices, scheduled maintenance, and standing authorizations.

**Outputs:** A response such as notifying the user, escalating to an analyst, stopping a process, or disabling an account.

**Method:** The model assesses authorization, explanations in the records, and evidence strength. Code uses those judgments to choose whether to close, queue, or act; further questions inform containment and escalation.

Source: [TypeSafe — Security Incidents](https://evals.typesafe.ai/security_incidents), accessed 2026-09-18. This describes TypeSafe's work, not our findings. See [[Use Cases#TypeSafe evaluation methodology]] for the shared comparison method and limitations.


## Reproducibility limitation

Complete reproduction materials, including the full evaluation dataset and runnable evaluation code, have not been found in the sources reviewed. This does not establish that reproduction is impossible; additional materials may become available or be provided on request.
