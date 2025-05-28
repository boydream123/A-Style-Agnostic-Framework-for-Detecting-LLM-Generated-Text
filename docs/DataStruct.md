# Dataset Dictionary Structure

Each sample in the dataset (i.e., each line in the output `*.jsonl` file) is a JSON object with the following fields:

| Key             | Data Type | Description                                                                                                | Example                                     |
|-----------------|-----------|------------------------------------------------------------------------------------------------------------|---------------------------------------------|
| `text`          | String    | The textual content of the sample. This can be either an original human-written text or an LLM-generated text. | "Once upon a time, in a land far away..." |
| `source`        | String    | Indicates the origin of the text. If it's an LLM-generated sample, `_llm` is appended to the original source name. | `"gutenberg"`, `"xsum"`, `"gutenberg_llm"`  |
| `label`         | Integer   | A binary label indicating whether the text is human-written or LLM-generated. `0` for human-written, `1` for LLM-generated. | `0`, `1`                                    |
| `original_text` | String    | (Optional) This field is present **only** if the sample is LLM-generated (`label: 1`). It contains the original human-written text that was provided to the LLM for rewriting. | "This is the original human sentence."      |

### Example JSONL entries:

**Human-written sample:**
```json
{"text": "The quick brown fox jumps over the lazy dog.", "source": "gutenberg", "label": 0}
```
**LLM-generated sample:**
```json
{"text": "In a realm of ancient lore, a swift, russet-furred creature did leap with graceful ease above a slumbering canine of indolent nature.", "source": "gutenberg_llm", "label": 1, "original_text": "The quick brown fox jumps over the lazy dog."}
```