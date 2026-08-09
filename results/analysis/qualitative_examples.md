# Qualitative and Error Analysis

All systems use the same ASSET source sentences and human references. SARI is shown per example.

## Automatic Diagnostic Flags

These are reproducible indicators for manual review. They are not treated as definitive factual errors.

| Flag | LSTM without attention | LSTM with attention | Qwen controlled three-shot |
| --- | ---: | ---: | ---: |
| Empty output | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| Unknown-token artifact | 330 (91.9%) | 276 (76.9%) | 0 (0.0%) |
| Separator artifact | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| Repeated word or phrase | 197 (54.9%) | 115 (32.0%) | 8 (2.2%) |
| Possible over-deletion | 36 (10.0%) | 14 (3.9%) | 0 (0.0%) |
| Possible under-simplification | 0 (0.0%) | 28 (7.8%) | 105 (29.2%) |
| Possible number loss | 106 (29.5%) | 48 (13.4%) | 3 (0.8%) |
| Number not found in source | 28 (7.8%) | 12 (3.3%) | 2 (0.6%) |
| Possible name loss | 301 (83.8%) | 227 (63.2%) | 35 (9.7%) |

## Overall Interpretation

Attention reduced unknown-token artifacts, repetition, over-deletion,
and loss of names and numbers compared with the no-attention model.
Qwen nearly eliminated these errors and produced the strongest overall
simplifications. However, Qwen was more likely to preserve complex
source wording instead of simplifying aggressively.

## Ten Representative Examples

### Example 1: Large attention advantage over the ablation

**Source:** Most of the songs were written by Richard M. Sherman and Robert B. Sherman.

**Human reference:** Many of the songs were penned by Richard M. Sherman and Robert B. Sherman.

| System | Output | SARI | Automatic flags |
| --- | --- | ---: | --- |
| LSTM without attention | most of the <unk> were <unk>, and the first single of the band were shown in the united states. | 15.96 | Unknown-token artifact, Possible name loss |
| LSTM with attention | most of the songs were written by richard m. sherman and robert b. sherman. | 60.85 | Possible under-simplification |
| Qwen controlled three-shot | Most of the songs were written by Richard M. Sherman and Robert B. Sherman. | 60.85 | Possible under-simplification |

**Manual assessment:** The no-attention model changes the meaning and adds unsupported information. The attention and Qwen outputs preserve the facts but mostly copy the source, so simplification is limited.

### Example 2: Large attention advantage over the ablation

**Source:** Both names became defunct in 2007 when they were merged into The National Museum of Scotland.

**Human reference:** Both names became unused when they joined The National Museum of Scotland.

| System | Output | SARI | Automatic flags |
| --- | --- | ---: | --- |
| LSTM without attention | the city of the university was founded in april 1046. | 17.47 | Possible number loss, Number not found in source, Possible name loss |
| LSTM with attention | both names became a <unk> in 2007 when they were merged into the national museum of scotland. | 56.50 | Unknown-token artifact |
| Qwen controlled three-shot | Both names became defunct in 2007 when they were merged into The National Museum of Scotland. | 57.82 | Possible under-simplification |

**Manual assessment:** The no-attention model produces unrelated information and an incorrect year. The attention model preserves the structure but loses the key term through an unknown token. Qwen preserves the facts but does not simplify the difficult word “defunct.”

### Example 3: Large attention advantage over the ablation

**Source:** He was appointed Companion of Honour (CH) in 1988.

**Human reference:** He was appointed Companion of Honour in 1988.

| System | Output | SARI | Automatic flags |
| --- | --- | ---: | --- |
| LSTM without attention | he was born in <unk>, new york city. | 20.92 | Unknown-token artifact, Possible number loss, Possible name loss |
| LSTM with attention | he was appointed companion of honour (ch) in 1988. | 58.18 | Possible under-simplification |
| Qwen controlled three-shot | He was appointed Companion of Honour (CH) in 1988. | 58.18 | Possible under-simplification |

**Manual assessment:** The no-attention model changes the original fact into an incorrect birthplace statement. The attention and Qwen outputs are accurate but make almost no simplification because the source is already short.

### Example 4: Large Qwen advantage over the attention model

**Source:** This leads to consumer confusion when 220 (1,048,576) bytes is referenced as 1 MB (megabyte) instead of 1 MiB.

**Human reference:** People get confused when the term MB is used instead of MiB.

| System | Output | SARI | Automatic flags |
| --- | --- | ---: | --- |
| LSTM without attention | this is also called the <unk>. | 18.06 | Unknown-token artifact, Possible over-deletion, Possible number loss, Possible name loss |
| LSTM with attention | this leads to consumer confusion when they are <unk>. | 15.37 | Unknown-token artifact, Possible number loss, Possible name loss |
| Qwen controlled three-shot | This causes confusion for consumers when 220 (1,048,576) bytes is referred to as 1 MB (megabyte) instead of 1 MiB. | 60.61 | None |

**Manual assessment:** Both LSTM models remove important technical details and numerical information. Qwen preserves the measurements and meaning while using clearer wording, making it the strongest output.

### Example 5: Large Qwen advantage over the attention model

**Source:** While at Kahn he was chief architect for the Fisher Building in 1928.

**Human reference:** In 1928, he worked at Kahn. He was the chief architect for the Fisher Building.

| System | Output | SARI | Automatic flags |
| --- | --- | ---: | --- |
| LSTM without attention | he was also a member of the <unk> <unk> <unk>. | 20.85 | Unknown-token artifact, Repeated word or phrase, Possible number loss, Possible name loss |
| LSTM with attention | while he was chief in the fisher of the world. | 23.32 | Possible number loss, Possible name loss |
| Qwen controlled three-shot | While at Kahn, he was the chief architect for the Fisher Building in 1928. | 65.40 | Possible under-simplification |

**Manual assessment:** The no-attention output is unrelated, while the attention output is ungrammatical and loses names and the date. Qwen is fluent and factually accurate, although its simplification is modest.

### Example 6: Large Qwen advantage over the attention model

**Source:** He also completed two collections of short stories entitled The Ribbajack & Other Curious Yarns and Seven Strange and Ghostly Tales.

**Human reference:** He also completed two short story collections. These were titled The Ribbajack & Other Curious Yarns and Seven Strange and Ghostly Tales.

| System | Output | SARI | Automatic flags |
| --- | --- | ---: | --- |
| LSTM without attention | he also wrote the <unk> <unk> and <unk> <unk>. | 15.43 | Unknown-token artifact, Repeated word or phrase, Possible name loss |
| LSTM with attention | he also completed two collections of short stories, the <unk> & other <unk> <unk> and seven strange and <unk> tales. | 26.32 | Unknown-token artifact, Repeated word or phrase, Possible name loss |
| Qwen controlled three-shot | He also completed two collections of short stories called The Ribbajack & Other Curious Yarns and Seven Strange and Ghostly Tales. | 66.32 | Possible under-simplification |

**Manual assessment:** Both LSTM outputs lose important names through unknown tokens. Qwen replaces “entitled” with the simpler word “called” while preserving both titles, although the sentence remains long.

### Example 7: Low attention-model SARI

**Source:** So far the 'celebrity' episodes have included Vic Reeves, Nancy Sorrell, Gaby Roslin, Scott Mills, Mark Chapman, Simon Gregson, Sue Cleaver, Carol Thatcher, Paul O'Grady and Lee Ryan.

**Human reference:** The 'celebrity' shows have included the following individuals: Vic Reeves, Nancy Sorrell, Gaby Roslin, Scott Mills, Mark Chapman, Simon Gregson, Sue Cleaver, Carol Thatcher, Paul O'Grady and Lee Ryan.

| System | Output | SARI | Automatic flags |
| --- | --- | ---: | --- |
| LSTM without attention | the movie stars peter <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk>, <unk> <unk> | 15.47 | Unknown-token artifact, Repeated word or phrase, Possible name loss |
| LSTM with attention | so far the ' <unk> ' episodes have included the reeves. | 14.34 | Unknown-token artifact, Possible over-deletion, Possible name loss |
| Qwen controlled three-shot | So far, the 'celebrity' episodes have included Vic Reeves, Nancy Sorrell, Gaby Roslin, Scott Mills, Mark Chapman, Simon Gregson, Sue Cleaver, Carol Thatcher, Paul O'Grady, and Lee Ryan. | 50.46 | Possible under-simplification |

**Manual assessment:** The no-attention model hallucinates information and repeats unknown tokens. The attention model deletes nearly all names. Qwen preserves every name but mostly copies the source, showing a trade-off between factual preservation and simplification.

### Example 8: Low attention-model SARI

**Source:** Benchmarking conducted by PassMark Software highlights the 2009 version's 52 second install time, 32 second scan time, and 7 MB memory utilization.

**Human reference:** Benchmarking done by PassMark Software highlights the 2009 version's 52 second install time, 32 second scan time, and 7 MB memory utilization.

| System | Output | SARI | Automatic flags |
| --- | --- | ---: | --- |
| LSTM without attention | the <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>, and <unk> <unk>. | 17.11 | Unknown-token artifact, Repeated word or phrase, Possible number loss, Possible name loss |
| LSTM with attention | <unk> conducted by <unk> software <unk> 2009 <unk> <unk> <unk> <unk> <unk>. | 14.84 | Unknown-token artifact, Repeated word or phrase, Possible number loss, Possible name loss |
| Qwen controlled three-shot | Benchmarking by PassMark Software shows that the 2009 version has an install time of 52 seconds, a scan time of 32 seconds, and uses 7 MB of memory. | 34.02 | None |

**Manual assessment:** Both LSTM models lose most technical details and numerical values. Qwen restructures the sentence clearly while preserving every measurement, making it the best output.

### Example 9: Low Qwen SARI

**Source:** The crosier symbolises the monasteries of the region.

**Human reference:** The crosier is a symbol of the local monasteries.

| System | Output | SARI | Automatic flags |
| --- | --- | ---: | --- |
| LSTM without attention | the <unk> <unk> is the <unk> of the <unk> <unk>. | 38.54 | Unknown-token artifact, Repeated word or phrase |
| LSTM with attention | the <unk> <unk> the <unk> of the region. | 42.35 | Unknown-token artifact, Repeated word or phrase |
| Qwen controlled three-shot | The crosier symbolises the monasteries in the region. | 28.92 | None |

**Manual assessment:** Both LSTM outputs are damaged by unknown tokens. Qwen preserves the meaning but performs almost no simplification and leaves difficult vocabulary unchanged, explaining its low SARI score.

### Example 10: Low Qwen SARI

**Source:** It is now the site of Meher Baba's samadhi (tomb-shrine) as well as facilities and accommodations for pilgrims.

**Human reference:** Now the place is Meher Baba's samadhi's tomb-shrine as well as facilities and housings for travelers.

| System | Output | SARI | Automatic flags |
| --- | --- | ---: | --- |
| LSTM without attention | it is the <unk> <unk>. | 23.50 | Unknown-token artifact, Repeated word or phrase, Possible over-deletion, Possible name loss |
| LSTM with attention | it is now the site of <unk> <unk> <unk> (<unk>) as well as facilities and <unk> for <unk>. | 32.13 | Unknown-token artifact, Repeated word or phrase, Possible name loss |
| Qwen controlled three-shot | It is now a site for Meher Baba's tomb-shrine as well as places for pilgrims to stay and visit. | 31.09 | None |

**Manual assessment:** The LSTM outputs lose essential names and meaning. Qwen is readable and preserves the main information despite its low SARI score, although “visit” may be a small unsupported addition.
