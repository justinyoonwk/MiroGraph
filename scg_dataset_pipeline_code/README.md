# SCG Dataset Pipeline Code

This folder turns extracted paper-level concepts into temporal Scientific Concept Graphs and prediction datasets.

Main roles:

- cluster and normalize extracted concept strings;
- map papers to final concept names;
- build temporal co-occurrence graphs with citation counts;
- generate transductive and inductive train/test candidate-pair files.

The public final datasets are stored in the repository-level `dataset/` folder.
