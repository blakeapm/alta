# ALTA (Active Learning for Text Analysis)

Active Learning for Text Analysis. ALTA provides a simple interface for document annotation using active learning. While implementing several active learning algorithms for text annotation, this package also facilitates random sampling approaches to document annotation.

## Sources

ALTA implements many of the algorithms discussed in:

*Miller, Blake, Fridolin Linder and Walter Mebane. ["Active Learning Approaches for Labeling Text: Review and Assessment of the Performance of Active Learning Approaches"](https://drive.google.com/file/d/1v2FEVjIIcVVldtk2P1WhgLkYRaaYAqVu/) Political Analysis, forthcoming. DOI: 10.1017/pan.2020.4*

## Current and Future Functionality

ALTA is currently in a minimum viable product stage of development and as such may change quite a bit over the next several weeks, so please keep track of any changes here.

### Algorithms currently implemented

- `textannotation_random`: Random annotation. Documents are sampled for annotation randomly.
- `textannotation_lasso`: An uncertainty sampling method using predicted probabilities from a lasso regression as a measure of uncertainty. Documents are sampled using absolute deviance of the models predicted probability from .5.

### Algorithms to be implemented in the future

- `textannotation_svm`: An uncertainty sampling method using a support vector machine (SVM) classifier. Documents are sampled using absolute distance from a the classifier's class-separating hyperplane.
- `textannotation_rf`: A "query by committee" method that uses the forest vote entropy from a random forest to select documents for annotation.

### Other future functionality

- `learning_curve`: Given the data frame output from `textannotation` models, plot a learning curve.
- `gen_error`: Provide an error report comparing the generalization error of randomly sampled data to actively sampled data to estimate bias in generalization error estimates coming from active learning.

## Installing ALTA

ALTA is not currently on CRAN, but can be installed from source on Github using the `devtools` package:

```r
# Install devtools from CRAN
install.packages("devtools")

# Install ALTA from GitHub:
devtools::install_github("blakeapm/alta")
```

## How to use ALTA

Load the alta package into your working directory.

```r
library(alta)
```

Load in a data frame with a column of document texts, a column with unique document ids,  and column(s) with annotations if some documents have already been annotated.

```r
hate_speech <- read.csv('hate_speech.csv', stringsAsFactors=FALSE)
```

Specify the text, id, and annotation column names. In the R console, you will be prompted to annotate documents one by one. A new augmented data frame will be assigned to `hate_speech_annotated`. This data frame will include all annotations made from console input (in the specified `y_col`) and will have new columns `batch`, and `active` which indicate the batch index at which the document was labeled and the algorithm used for document sampling: random (FALSE) or active (TRUE).

```r
hate_speech_annotated <- textannotation_lasso(hate_speech, 
                                              text_col='comment_text',
                                              y_col='obscene',
                                              doc_id_col='id')
```

ALTA defaults to 10 batches and 10 documents per batch (a total of 100 documents to be labeled in each session). Larger batches help to limit the time spent fitting text models used to sample documents. If you face fewer computational resource constraints, you could reduce batch size to 1 and increase the number of batches to 100. This would also result in 100 documents labeled per session, but a new model would be fit after annotating each document.

```r
hate_speech_annotated <- textannotation_lasso(hate_speech, 
                                              text_col='comment_text',
                                              y_col='obscene',
                                              doc_id_col='id',
                                              n_batches=100,
                                              batch_size=1)
```
