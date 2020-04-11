#' Text Annotation Using Uncertainty Sampling via Lasso Regression
#'
#' This function supports annotation of documents using an uncertainty sampling approach using predicted response of lasso regression as an uncertainty measure. This algorithm samples a batch of unlabeled documents with a predicted probability nearest to .5 for annotation via the R console. This function returns a data frame with annotations and their resepective batch_id and a boolean flag for whether or not the documents were sampled actively (or randomly).
#' @param data A dataframe with a 'text' column and 'doc_id' column.
#' @param text_col A string with the name of the column containing document 
#' text. Defaults to 'text'
#' @param y_col A string with the name of the column containing document 
#' text. Defaults to 'y'
#' @param doc_id_col A string with the name of the column containing document
#'  ids. Defaults to 'doc_id'
#' @param batch_size An integer for the number of documents to sample at 
#' each iteration of the active learning algorithm. Defaults to 10.
#' @param n_batches An integer for the number of total batches to label.
#' Defaults to 10..
#' @param n_cores An integer for the number of cores to use in the estimation 
#' of the lasso model.
#' @param rand_minority_min_n An integer for the minimum number of documents
#' in the minority class needed to begin active learning. If the minority
#' class has fewer observations than `rand_minority_min_n`, documents will
#' be sampled randomly. Defaults to 100.
#' @return A new data frame object derived from the data frame passed into
#' the the 'data' argument. This new data frame includes new annotations from
#' the annotation session in the column specified in the 'y_col' argument. 
#' A new column 'batch' has been added to specify the batch at which each 
#' new document was labeled, and a new column 'active' has been added to 
#' specify  whether a newly-labeled document was sampled actively or 
#' passively (randomly).
#' @import quanteda glmnet doMC
#' @author Blake Miller
#' @keywords active learning text supervised learning annotation
#' @export
#' @examples
#' data <- read.csv('wikipedia_hate_speech.csv', stringsAsFactors=FALSE)
#' data_annotated <- textannotation_lasso(data, text_col='comment_text', y_col='toxic', doc_id_col='id', n_batches=100, batch_size=5)

textannotation_lasso <- function(data, text_col='text', y_col='y', doc_id_col='doc_id', n_batches=10, batch_size=10, rand_minority_min_n=100, n_cores=2) {
	if (!y_col %in% names(data)) {
		warning(paste("No existing y column found in corpus data frame. Creating new column: '", y_col, "\'", sep=""))
		data[y_col] <- NA
	}
	if (!'active' %in% names(data)) {
		data$active <- NA
	}
	if (!'batch_id' %in% names(data)) {
		data$batch_id <- NA
		batch_id <- 0
	} else {
		batch_id <- max(data$batch_id)
	}
	for (i in 1:n_batches) {
		batch_id <- batch_id + 1
		cor <- corpus(data, text_field='comment_text')
		dfm <- dfm(cor, remove=stopwords("english"), verbose=TRUE)
		dfm <- dfm_trim(dfm, min_docfreq = 2, verbose=TRUE)
		labeled <- data[which(!is.na(data[y_col])),]
		unlabeled <- data[which(is.na(data[y_col])),]
		message(paste("\nStarting batch ", batch_id, "/", batch_size, "...\n\nTotal labeled docs: ", nrow(labeled), "/", nrow(data), "\n", sep=""))
		if (length(labeled) == nrow(data)) {
			stop(paste("All documents are already labeled for the specified column: '", y_col, "\'", sep=""))
		}
		if (nrow(labeled) > rand_minority_min_n - 1) {
			is_active <- TRUE
			x_l <- dfm_subset(dfm, !is.na(docvars(dfm, y_col)))
			x_u <- dfm_subset(dfm, is.na(docvars(dfm, y_col)))
			if (batch_id == 1) {
				message("\nTraining lasso model...\n", sep="")
			} else {
				message("\nRetraining lasso model...\n", sep="")
			}
			registerDoMC(cores=n_cores)
			mod <- cv.glmnet(x_l, docvars(x_l, y_col), family="binomial", alpha=1, nfolds=5, parallel=TRUE, intercept=TRUE)
			pred <- predict(mod, x_u, s="lambda.1se", type="response")
			#mod <- svm(y ~ x_l, x_u, cost = 10, scale = FALSE, kernel = 'linear')
			#pred <- predict(mod, decision.value=T)
			sorted <- sort(abs(pred - .5), decreasing=FALSE, index.return=TRUE)
			to_label <- unlabeled[sorted$ix[1:batch_size],]
			to_label$score <- sorted$x[1:batch_size]
		} else {
			is_active <- FALSE
			warning("Not enough labeled observations in the minority class. Randomly sampling...")
			to_label <- unlabeled[sample(1:nrow(unlabeled), batch_size), ]
		}
		hrule <- "\n——————————————————————————————————————————————"
		for (i in 1:nrow(to_label)) {
			id <- to_label[i,doc_id_col]
			text <- to_label[i,text_col]
			score <- to_label[i,"score"]
			message(paste(hrule, "\nDocument ID: ", id, "\n\n", "Score: ", score, "\n\n", "Document Text: ", "\n\n", text, hrule, sep=''))
			while (TRUE) {
				label <- readline(prompt=paste("\n\nEnter label for '", y_col, "' (0, 1): ", sep=""))
				if (grepl("[0-1]", label)) {
					break
				} else {
					warning("Invalid input. You must enter either 0 for false or 1 for true.")
				}
			}
			data[data[doc_id_col] == id, y_col] <- label
			data[data[doc_id_col] == id, 'batch_id'] <- batch_id
			data[data[doc_id_col] == id, 'active'] <- is_active
		}
		message(paste(hrule, "\n\nDone with batch ", batch_id, "/", batch_size, ".\n\nRefitting model with annotations from the previous batch...\n\n", sep=""))
	}
	data
}