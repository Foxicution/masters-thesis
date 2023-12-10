<!--- TODO: finish the usage section -->
## Usage
```
# 1. get github data for repositories under 20000 stars (because of api limits)
python mt/github_data.py
# 2. featurize repos on a file level (filtering for repos that fit applied)
python mt/pull_repos.py
# 3. put features into the same files
python mt/process_features.py
# 4. clean the features and put them to the final file
python mt/clean_features.py
# 5. train model and predict feature importance
python mt/prediction_feat_importance.py
# 6. (optional) exploratory analysis
python mt/analysis.py
```

## About


## TODO
- [ ] Do population weighting for the number of issues. Take features like lines of code, number of contributors, etc. into account.
- [X] Grab data from each of the blobs inside a commit.
- [X] Featurize each of the blobs into trees.
- [X] Save the data to memory for each commit.
- [ ] Update the thesis and presentation
Optional:
- [ ] Take not only file level features but repo level features using tree-sitter-graph and stack-graphs (HARD)
- [ ] Write the about me section



## Useful links
https://github.com/IBM/tree-sitter-codeviews
https://tree-sitter.github.io/tree-sitter/syntax-highlighting#basics
https://github.com/github/stack-graphs/tree/main/tree-sitter-stack-graphs/examples
https://github.com/tree-sitter/tree-sitter-graph
https://github.blog/2021-12-09-introducing-stack-graphs/
https://drops.dagstuhl.de/opus/volltexte/2023/17778/pdf/OASIcs-EVCS-2023-8.pdf
https://www.youtube.com/watch?v=zz3A3Rv2PHk

https://docs.rs/tree-sitter-graph/0.10.4/tree_sitter_graph/reference/ (maybe could use this with a lot of effort)
Search for code ai models

https://github.com/salesforce/CodeT5
https://vectara.com/large-language-models-llms-for-code-generation-part-1/
https://huggingface.co/blog/starcoder
https://huggingface.co/bigcode
https://arxiv.org/abs/2305.07922
