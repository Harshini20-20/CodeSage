# CodeSage Phase 2D and 2E

## Phase 2D — Relationship-aware answers

The RAG pipeline builds bounded relationship context from the repository graph
and retrieved source locations. Confirmed calls and inheritance are labeled as
confirmed, unresolved calls remain unresolved, and the prompt tells the model
not to infer relationships from co-occurrence. Missing graph context falls back
to an explicit no-evidence message. Existing custom prompt builders keep their
two-argument interface.

## Phase 2E — Evidence-traceable answers

The prompt asks the model to cite retrieved context blocks using `[SOURCE N]`
labels and to use file/line metadata only when present. After generation, the
pipeline validates explicit source labels against the actual retrieved source
count. Invalid labels are removed and an evidence note is appended; the result
also exposes `invalid_source_references` for programmatic inspection.

The citation validator checks reference integrity only. It does not prove that
a cited source semantically supports a claim, and it does not fabricate
citations for uncited statements. Source metadata remains available through
`RAGResult.sources`, `source_files`, and the existing Streamlit source display.

## Compatibility

No new runtime dependencies are introduced. Existing retrieval and RAG behavior
without a graph remains supported. This change does not alter configured Python,
Ollama, or embedding model versions.
