
## Evaluating Neural Multi-Field Document Representations for Patent Classification
This is the companion code for the experiments reported in the paper "Evaluating Neural Multi-Field Document Representations for Patent Classification"  by 
Subhash Chandra Pujari, Fryderyk Mantiuk, Mark Giereth, Jannik Strötgen and Annemarie Friedrich published in BIR 2022 colocated with ECIR 2022.
The code allows the users to reproduce the results reported in the paper and extend the model to new datasets. 
For any queries regarding code or dataset, you can contact [Subhash Pujari](subhashchandra.pujari@de.bosch.com). 
Please cite the paper when reporting, reproducing or extending the results as:
## Citation
```
@inproceedings{pujari-2022-bir,
    author = {
        Subhash Chandra Pujari and 
        Fryderyk Mantiuk and 
        Mark Giereth and 
        Jannik Str{\"{o}}tgen and 
        Annemarie Friedrich
    },
  title     = {Evaluating Neural Multi-Field Document Representations for Patent Classification},
  booktitle = {BIR 2022: 12th International Workshop on Bibliometric-enhanced Information Retrieval at ECIR 2022, April 10, 2022, hybrid},
  month     = {April},
  year      = {2022},
}
```


## Purpose of the project
This software is a research prototype, solely developed for and published as part of the publication 
"Evaluating Neural Multi-Field Document Representations for Patent Classification", BIR 2022 at ECIR 2022. It will
neither be maintained nor monitored in any way.

## Setup
* pip install -r requirements.txt
* update the config file to provide directory paths and model parameters. Sample config can be found in ./config/ folder.
* cd bir2022-neural-patent-classification/patent_classification
* python train.py --cfg path-to-config-file --mode <debug/dev>
* python predict.py --cfg path-to-config-file --exp_dir path-to-exp-dir

## License
The code is open-sourced under the AGPL-3.0 license. See the [LICENSE](LICENSE) file for details.

A data set sampled from USPTO coprus is located in the folder uspto-70k-enriched are
licensed under a [Creative Commons Attribution 4.0 International License] (http://creativecommons.org/licenses/by/4.0/) (CC-BY-4.0).


## USPTO-7k-enriched Dataset

The patent classification task falls under the category of hierarchical multi-label classification. 
A patent document contains `title`, `abstract`, `claims` and `description` as four textual fields. 
Because of the large text, most of the previous work focused on title, abstract and claims as patent fields.
In our paper [1], we make use description as a more elaborate patent field.
For evaluation, we create a new dataset (USPTO-70k-enriched) from the previously releasd USPTO-70k dataset 
which contains title and abstract as patent fields.
Now, the dataset is enriched with four additional text columns, claims, brief-summary, fig-desc, detail-desc, 
where the later three columns are the subfield of description. Both the datasets are created from the bulk-data-dump 
provided by United States Patent and Trademark Office (USPTO) released under CC-BY-4.0.
We also release the dataset under the same license, CC-BY-4.0.


The dataset could be downloaded from the following link: https://zenodo.org/record/6992298

```
./bir_dataset_2022/
    train.csv
    dev.csv
    test.csv 
```


Each row in these CSV files represents an entry corresponding to a patent. 
Each dataset file contains eight columns.
The column represents the value corresponding to patent fields (`title`, `abstract`, `claims`, `description`). 
The text within `description` field is further divided into three subfields, `brief-summary`, `detail-desc` and `fig-desc` as subfields.
The `patent_id` column uniquely identifies a patent family.
The label column lists a set of labels from the third level from the CPC taxonomy that were assigned to a patent.
As shown in Table 1, a document might be associated with more than one label, thus making the classification to target categories 
a multi-label document classification task.

* `patent_id`
* `title`
* `abstract`
* `claims`
* `brief-summary`
* `fig-desc`
* `detail-desc`
* `labels`


Table 1: Dataset Statistics.

|  dataset       | instances        | total labels   | Average labels per Instance|
|----------------|------------------|----------------|----------------------------|
|train           |          50625   |      630       |            1.98            |
|dev             |          10000   |      575       |            2.25            |
|test            |         10000    |      573       |            2.32            |


## References

[1] Subhash Chandra Pujari, Fryderyk Mantiuk, Mark Giereth, Jannik Strötgen, and Annemarie Friedrich. 2022. 
Evaluating Neural Multi-Field Document Representations for Patent Classification. 
In *Proceedings of the 12th International Workshop on Bibliometric enhanced Information Retrieval (BIR’22) 
co-located with the 44th European Conference on Information Retrieval (ECIR’22)*, Stavanger, Norway. CEUR-WS.org.

[2] Subhash Chandra Pujari, Annemarie Friedrich, and Jannik Strötgen. 2021. 
A Multi-Task Approach to Neural Multi-Label Hierarchical Patent Classification using Transformers. 
In *Proceedings of the 43rd European Conference on Information Retrieval (ECIR’21)*, Online.