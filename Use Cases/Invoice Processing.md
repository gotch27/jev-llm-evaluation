# Invoice Processing

[[Use Cases|← Use cases]]

Added on 2026-09-18 to document TypeSafe’s published evaluation as background for the thesis. Any independent experiments on a similar task will be documented in a separate note.

## TypeSafe's published evaluation

**Task:** Decide whether an invoice can be paid and what must happen first.

**Inputs:** The invoice, purchase order, contract, vendor record, prior invoices, correspondence, delivery evidence, and approvals.

**Outputs:** One or more actions such as paying, scheduling payment, holding for documents, requesting corrections, disputing items, or requesting fraud review.

**Method:** Model questions assess the documents while code handles calculations and rules. The workflow checks reasons to stop, collects holds and disputes, and determines the conditions for releasing payment.

Source: [TypeSafe — Invoice Processing](https://evals.typesafe.ai/invoice_processing), accessed 2026-09-18. This describes TypeSafe's work, not our findings. See [[Use Cases#TypeSafe evaluation methodology]] for the shared comparison method and limitations.


## Reproducibility limitation

Complete reproduction materials, including the full evaluation dataset and runnable evaluation code, have not been found in the sources reviewed. This does not establish that reproduction is impossible; additional materials may become available or be provided on request.
