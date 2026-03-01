# DuckDB NER Extension Development Task

Your task is to create a DuckDB community extension for Named Entity Recognition
(NER) written in C++.

The goal is to create an extension that can be loaded and used seamlessly within
SQL queries.

## Expected UX and API

The extension must support the following usage:

```sql
INSTALL ner; 
LOAD ner;

-- Default usage (uses the bundled/default lightweight model)
-- 'truncate=true' instructs the extension to silently truncate strings that exceed the model's token limit.
SELECT *, ner("column_storing_strings", truncate=true) as entities
FROM "table_name";

-- Output Format:
-- The `ner()` function MUST return a DuckDB LIST of STRUCTs. 
-- Example: [{'entity': 'DuckDB Labs', 'label': 'ORG'}, {'entity': 'Amsterdam', 'label': 'LOC'}]

-- Loading Custom Models:
-- Users should be able to configure a custom model path via a DuckDB setting or PRAGMA before querying.
SET ner_model_path = '/path/to/custom/model.onnx';
SELECT ner("column_storing_strings") FROM "table_name";
```

## Core Requirements

- Lightweight & Performant: The extension must bundle (or automatically fetch on
  first use) a lightweight, performant NER model.

- C++ Ecosystem: Because DuckDB extensions are statically linked C++, you must
  carefully select a C++ inference engine (e.g., ggml, ONNX Runtime, or a pure
  C++ CRF/statistical model) that plays nicely with DuckDB's cross-compilation
  matrix.

- Custom Models: Easily allow the user to point the extension to a model they
  previously downloaded on their local machine.

1. Planification

First, you must plan the work. Do not write the implementation code until the
plan is defined.

- Template Research: This repo has been started from the official DuckDB
  extension template. Read the README thoroughly and all relevant pages linked
  in it.

- Community Ecosystem: Check other DuckDB community extensions to understand how
  they modified the template (especially CMakeLists.txt) to manage heavy
  external dependencies or cross-compilation. List of extensions:
  https://duckdb.org/community_extensions/list_of_extensions

- CI/CD Constraints: This extension won't be accepted unless ALL GitHub Actions
  work (Linux, macOS, Windows, WebAssembly, etc.). Make sure to check the
  workflows, jobs, and actions to clearly understand architectural constraints.

- Architecture Decision: Propose exactly how the C++ inference engine will be
  integrated and how the default model weights will be distributed (bundled in
  the binary vs. downloaded to a local cache).

Write your plan in a PLAN.md file in this repo. Keep track of your progress in
it.

If a PLAN.md is already in there, it means it has already been approved. Check
where things were left and keep working.

2. Development

Once the plan is accepted, proceed with development.

- Testing: Create thorough tests to make sure the extension works as expected.
  You must test accuracy (are entities extracted correctly?) and performance
  (throughput on large tables, memory leaks).

- Documentation: Describe the benchmark and test results in a TEST.md file.

- Iterative Commits: Every time you reach a milestone, stage and commit your
  changes. Ensure the GitHub actions, workflows, and jobs are passing before
  moving to the next milestone.
