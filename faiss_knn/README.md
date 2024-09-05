## Run book

1. Run the train index command

`python run.py --command train_index --config cc_config.yaml --xb mini_cc`


3. Run the index-shard command so it produces sharded indexes, required for the search step

`python run.py --command index_shard --config cc_config.yaml --xb mini_cc`


6. Send jobs to the cluster to run search

`python run.py --command search --config cc_config.yaml --xb mini_cc`


Remarks about the `search` command: it is assumed that the database vectors are the query vectors when performing the search step.
a. If the query vectors are different than the database vectors, it should be passed in the xq argument
b. A new dataset needs to be prepared (step 1) before passing it to the query vectors argument `–xq`

`python run.py --command search --config cc_config.yaml --xb mini_cc --xq <QUERIES_DATASET_NAME>`


6. We can always run the consistency-check for sanity checks!

`python run.py --command consistency_check --config cc_config.yaml --xb mini_cc`

