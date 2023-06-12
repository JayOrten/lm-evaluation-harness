# New Task Guide

`lm-evaluation-harness` is a framework that strives to support a wide range of zero- and few-shot evaluation tasks on autoregressive language models (LMs). 

This documentation page provides a walkthrough to get started creating your own task.

## Setup

If you haven't already, go ahead and fork the main repo, clone it, create a branch with the name of your task, and install the project requirements in your environment:

```sh
# After forking...
git clone https://github.com/<YOUR-USERNAME>/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout -b <task-name>
pip install -e ".[dev]"
```

As a concrete example, we'll walk through reimplementing the `gsm8k` benchmark (a *generative* task which requires sampling text from a model) and the `sciq` benchmark. (a *discriminative*, or *multiple choice*, task where the model picks the most likely of several fixed answer choices).

## Creating a YAML file

- Tasks in eval harness are largely implemented via YAML files.

- mention the tasks worth "forking"/building off of

- Step through the different args all tasks will need

To implement a new standard task, we'll need to write a YAML file which configures our task logic. We start by making a new empty YAML file:

```sh
touch lm_eval/tasks/new_mcqa.yaml
```
or
```sh
touch lm_eval/tasks/new_generative_task.yaml
```

### Selecting and configuring a dataset

All data downloading and management is handled through the HuggingFace (**HF**) [`datasets`](https://github.com/huggingface/datasets) API. So, the first thing you should do is check to see if your task's dataset is already provided in their catalog [here](https://huggingface.co/datasets). If it's not in there, please consider adding it to their Hub to make it accessible to a wider user base by following their [new dataset guide](https://github.com/huggingface/datasets/blob/master/ADD_NEW_DATASET.md)
.

Once you have a HuggingFace dataset prepared for your task, we want to assign our new YAML to use this dataset:

```yaml
dataset_path: ... # the name of the dataset on the HF Hub.
dataset_name: ... # the dataset configuration to use. Leave `null` if your dataset does not require a config to be passed. See https://huggingface.co/docs/datasets/load_hub#configurations for more info.
dataset_kwargs: null # any extra keyword arguments that should be passed to the dataset constructor, e.g. `data_dir`.
```

Next, we'd like to tell our task what the dataset's train, validation, and test splits are named, if they exist:

```yaml
training_split: <split name of training set, or `null`>
validation_split: <split name of val. set, or `null`>
test_split: <split name of test set, or `null`>
```
Tests will run on the `test_split` if it is available, and otherwise evaluate on the `validation_split`.

We can also specify from which split the task should retrieve few-shot examples via:
```yaml
fewshot_split: <split name to draw fewshot examples from, or `null`>
```
though if this is not set, we will default to train/validation/test sets, in that order.

### Writing a prompt

The next thing we need to do is decide what format to use when presenting the data to the LM. This is our **prompt**, where we'll define both an input and output format.

We support the [Jinja 2](https://jinja.palletsprojects.com/en/3.1.x/) templating language for writing prompts. In practice, this means you can take your dataset's columns and do many basic string manipulations to place each document into prompted format.

To write a prompt, users are required to write two YAML fields in Jinja as strings:
```yaml
doc_to_text:
doc_to_target:
```
Suppose our dataset has a `"question"` field, and an `"answer"` field, which are both strings. We want the model to see, if given a `document` object that is a row of our dataset:
```
Question: {document[question]}
Answer:
```
We do this by writing 
```yaml
doc_to_text: "Question: {{question}}\nAnswer:"
```
Such that {{question}} will be replaced by `doc["question"]` when rendering the prompt template.

Our intended output is for the model to predict a single whitespace, and then the answer to the question. We do this via:
```yaml
doc_to_target: "{{answer}}"
```

**Important**: We always add one whitespace between the input and output, such that the full input-output string is `doc_to_target(doc) + " " + doc_to_text(doc)`. doc_to_text and doc_to_target should not contain trailing right or left whitespace, respectively.

TODO: mention promptsource here, or reserve it for advanced guide

#### Multiple choice format

- template_aliases

- expected mcqa setup

### Setting metrics

You're almost done! Now we need to choose how to score our task.
- *If this is a multiple choice task:* do you just want to check your model's accuracy in choosing the correct answer choice? 
- *If this is a generation task:* do you just want to check how often your model outputs *exactly the ground-truth output string provided*?

If the answer to the above is no: you'll need to record what scoring metrics to use! Metrics can be listed in the following format:

```yaml
metric_list:
  - metric: <name of the metric here>
    aggregation: <name of the aggregation fn here>
    higher_is_better: <true or false>
  - metric: ...
    aggregation: ...
    higher_is_better: ...
```

For a full list of natively supported metrics and aggregation functions see `TODO: we should list out all supported metrics, aggregations, models, somewhere in the docs.` All metrics supported in [HuggingFace Evaluate](https://github.com/huggingface/evaluate/tree/main/metrics) can also be used, and will be loaded if a given metric name is not one natively supported in `lm-eval`.

### Optional, more advanced setup

Some tasks may require more advanced processing logic than is described in this guide.

As a heuristic check:
* Does your task require generating multiple free-form outputs per input document?
* Does your task require complex, multi-step post-processing of generated model outputs?
* Does your task require subsetting documents on the fly based on their content?
* Do you expect to compute metrics after applying multiple such processing steps on your model outputs?
* Does your task rely on metrics that need a custom implementation? 

For more detail on the task system and advanced features, see `docs/advanced_task_guide.md` . If none of the above sound like they apply to your task, it's time to continue onto checking your task performance!

### Task name + groups (registering a task)

To test a task conveniently, it helps to *register* the task--that is, to give it a name and make the `lm-eval` library aware it exists!

If you're writing your YAML file inside the `lm_eval/tasks` folder, you just need to give your task a name! You can do this inside your YAML file:

```yaml
task: <name of the task>
```
Including a task name is mandatory.

It is often also convenient to label your task with several `groups`, or tags, though this field is optional:

```yaml
group:
  - group1
  - group2
```
This will add your task to the `group1` and `group2` groups, enabling people to know how to categorize your task, and if desired run all tasks in one of these groups at once, your task along with them.


If your task is not in the `lm_eval/tasks` folder, you'll need to tell the Eval Harness where to look for YAML files.

You can do this via adding the Python snippet 

```python
from lm_eval.tasks import include_task_folder
include_task_folder("/path/to/yaml/parent/folder")
```
to the top of any Python file that is run or imported when performing evaluation, such as `main.py`.

Passing `--tasks /path/to/yaml/file` is also accepted.


## Checking validity

- write_out

## Checking performance ; implementation equivalence

## Task impl. checklist

- turn this into a GH PR template too

- README.md in task dir

## Submitting your task

You're all set! Now push your work and make a pull request! Thanks for the contribution 👍. If there are any questions, please leave a message in the `#lm-thunderdome` channel on the EAI discord!