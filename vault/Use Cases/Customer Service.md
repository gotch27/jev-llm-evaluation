# Customer Service

[← Use cases](../Use%20Cases.md)

Added on 2026-09-18 to document TypeSafe’s published evaluation as background for the thesis. Any independent experiments on a similar task will be documented in a separate note.

## TypeSafe's published evaluation

**Task:** Choose the next actions for a support assistant during a customer conversation.

**Inputs:** Conversation history, customer records, account details, and a pending proposal where relevant.

**Outputs:** Actions such as responding, refunding, freezing a card, handing off, flagging for review, or closing.

**Method:** The model assesses intent, frustration, urgency, and risk. Conditional follow-ups examine consent and account issues. The workflow also checks the assistant’s claims against records before code selects actions.

Source: [TypeSafe — Customer Service](https://evals.typesafe.ai/customer_service), accessed 2026-09-18. This describes TypeSafe's work, not our findings. See [Use Cases — TypeSafe evaluation methodology](../Use%20Cases.md#typesafe-evaluation-methodology) for the shared comparison method and limitations.


## Reproducibility limitation

Complete reproduction materials, including the full evaluation dataset and runnable evaluation code, have not been found in the sources reviewed. This does not establish that reproduction is impossible; additional materials may become available or be provided on request.
