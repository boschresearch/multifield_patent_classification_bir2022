# USPTO-7k-enriched Dataset

The patent classification task falls under the category of hierarchical multi-label classification. 
A patent document contains `title`, `abstract`, `claims` and `description` as four textual fields. 
Because of the large text, most of the previous work focused on title, abstract and claims as patent fields.
In our paper [1], we make use description as a more elaborate patent field.
For evaluation, we create a new dataset (USPTO-70k-enriched) from the previously releasd USPTO-70k dataset 
which contains title and abstract as patent fields (Work on reference **RBGA-4320987**).
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