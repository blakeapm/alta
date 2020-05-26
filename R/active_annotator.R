sample_rf <- function(dfm, y_col, batch_size, batch_id, n_cores) {
	calc_entropy <- function(x) {
		freq <- table(x)/length(x)
		-sum(freq * log2(freq))
	}
	x_l <- dfm_subset(dfm, !is.na(docvars(dfm, y_col)))
	x_u <- dfm_subset(dfm, is.na(docvars(dfm, y_col)))
	if (batch_id == 1) {
		message("\nTraining random forests model...\n")
	} else {
		message("\nRetraining random forests model...\n")
	}
	if (is.null(n_cores)) {
		mod <- ranger(
			x = x_l, 
			y = docvars(x_l, y_col), 
			num.trees = 500,
			mtry = floor(sqrt(dim(x_l)[2])),
			classification = TRUE,
		)
	} else {
		mod <- ranger(
			x = x_l, 
			y = docvars(x_l, y_col), 
			num.trees = 500,
			mtry = floor(sqrt(dim(x_l)[2])),
			classification = TRUE,
			num.threads = n_cores
		)
	}
	pred <- predict(mod, x_u, predict.all=TRUE)
	pred <- apply(pred$predictions, 1, calc_entropy)
	sorted <- sort(pred, decreasing=TRUE, index.return=TRUE)
	sorted
}

sample_lasso <- function(dfm, y_col, batch_size, batch_id, n_cores) {
	x_l <- dfm_subset(dfm, !is.na(docvars(dfm, y_col)))
	x_u <- dfm_subset(dfm, is.na(docvars(dfm, y_col)))
	if (batch_id == 1) {
		message("\nTraining lasso model...\n")
	} else {
		message("\nRetraining lasso model...\n")
	}
	if (is.null(n_cores)) {
		registerDoMC(cores=n_cores)
	} else {
		registerDoMC()
	}
	mod <- cv.glmnet(x_l, docvars(x_l, y_col), family="binomial", alpha=1, nfolds=5, parallel=TRUE, intercept=TRUE)
	pred <- predict(mod, x_u, s="lambda.1se", type="response")
	sorted <- sort(abs(pred - .5), decreasing=FALSE, index.return=TRUE)
	sorted
}

sample_svm <- function(dfm, y_col, batch_size, batch_id, n_cores) {
	x_l <- dfm_subset(dfm, !is.na(docvars(dfm, y_col)))
	x_u <- dfm_subset(dfm, is.na(docvars(dfm, y_col)))
	if (batch_id == 1) {
		message("\nTraining svm model...\n")
	} else {
		message("\nRetraining svm model...\n")
	}
	y_l <- factor(docvars(x_l, y_col))
	# Tune C
	message("Tuning C using 5-fold cross-validation...\n")
	if (is.null(n_cores)) {
		registerDoMC(cores=n_cores)
	} else {
		registerDoMC()
	}
	folds <- sample(1:5, nrow(x_l), replace=TRUE)
	Cs <- c(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500)
	perf <- foreach(i = 1:10) %dopar% {
		fs <- numeric(5)
		for (j in 1:5) {
			mod <- svm(x = x_l[folds != j,], y = y_l[folds != j], cost=Cs[i], kernel="linear", type = 'C-classification') 
			pred <- predict(mod, x_l[folds == j,])
			cm <- table(pred, y_l[folds == j])
			tp <- cm[1,1]
			fp <- sum(cm[1,]) - tp
			fn <- sum(cm[,1]) - tp
			p <- tp / (tp + fp)
			r <- tp / (tp + fn)
			fs[j] <- 2 * (p * r)/(p + r)
		}
		mean(fs)
	}
	C <- Cs[which.max(unlist(perf))]
	mod <- svm(x = x_l, y = y_l, cost=C, kernel="linear", type = 'C-classification') 
	pred <- attr(predict(mod, x_u, decision.values=TRUE), "decision.values")
	sorted <- sort(abs(pred), decreasing=FALSE, index.return=TRUE)
	sorted
}

#' Text Annotation Using Active Learning
#'
#' This function provides a simple R console interface for annotation of documents using active learning. It can sample texts for annotation using a variety of active learning methods. This function returns a data frame with annotations and their resepective 'batch_id' and a boolean flag 'active' to identify which batches were sampled actively. This flag helps identify randomly-sampled data which can provide an unbiased estimate of generalization error.
#' @param data A dataframe with a 'text' column and 'doc_id' column.
#' @param text_col A string with the name of the column containing document 
#' text. Defaults to 'text'
#' @param y_col A string with the name of the column containing document 
#' text. Defaults to 'y'
#' @param doc_id_col A string with the name of the column containing document
#'  ids. Defaults to 'doc_id'
#' @param method 'lasso' samples unlabeled documents with a predicted 
#' response nearest to .5 from a lasso regression classifier. 'svm' samples 
#' unlabeled documents closest to the class-separating hyperplane of a 
#' support vector machine. 'rf' samples unlabeled documents based on the
#' vote entropy of all trees in a random forest classifier.
#' @param n_batches An integer for the number of total batches to label.
#' Defaults to 10..
#' @param batch_size An integer for the number of documents to sample at 
#' each iteration of the active learning algorithm. Defaults to 10.
#' @param rand_minority_min_n An integer for the minimum number of documents
#' in the minority class needed to begin active learning. If the minority
#' class has fewer observations than `rand_minority_min_n`, documents will
#' be sampled randomly. Defaults to 100.
#' @param n_cores An integer for the number of cores to use in the estimation 
#' of the model.
#' @return A new data frame object derived from the data frame passed into
#' the the 'data' argument. This new data frame includes new annotations from
#' the annotation session in the column specified in the 'y_col' argument. 
#' A new column 'batch' has been added to specify the batch at which each 
#' new document was labeled, and a new column 'active' has been added to 
#' specify  whether a newly-labeled document was sampled actively or 
#' passively (randomly).
#' @import quanteda glmnet ranger e1071 doMC foreach
#' @author Blake Miller
#' @keywords active learning text supervised learning annotation
#' @export
#' @examples
#' data <- read.csv('../example_data/hate_speech.csv', stringsAsFactors=FALSE)
#' data_annotated <- active_annotator(data, text_col='comment_text', y_col='obscene', doc_id_col='id', n_batches=10, batch_size=5, method='lasso')
#'
active_annotator <- function(data, dfm=NULL, text_col='text', y_col='y', doc_id_col='doc_id', method='lasso', n_batches=10, batch_size=10, rand_minority_min_n=100, n_cores=NULL) {
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
		if (is.na(batch_id)) {
			batch_id <- 1
		}
	}
	for (i in 1:n_batches) {
		batch_id <- batch_id + 1
		if (is.null(dfm)) {		
			cor <- corpus(data, text_field=text_col)
			dfm <- dfm(cor, remove=stopwords("english"), verbose=TRUE)
			dfm <- dfm_trim(dfm, min_docfreq = 2, verbose=TRUE)
		}
		labeled <- data[which(!is.na(data[y_col])),]
		unlabeled <- data[which(is.na(data[y_col])),]
		message(paste("\nStarting batch ", i, "/", n_batches, "...\n\nTotal labeled docs: ", nrow(labeled), "/", nrow(data), "\n", sep=""))
		if (length(labeled) == nrow(data)) {
			stop(paste("All documents are already labeled for the specified column: '", y_col, "\'", sep=""))
		}
		if (nrow(labeled) > rand_minority_min_n - 1) {
			is_active <- TRUE
			if (method == 'lasso') {
				ids <- sample_lasso(dfm, y_col, batch_size, batch_id, n_cores)
			} else if (method == 'rf') {
				ids <- sample_rf(dfm, y_col, batch_size, batch_id, n_cores)
			} else if (method == 'svm') {
				ids <- sample_svm(dfm, y_col, batch_size, batch_id, n_cores)
			} else {
				stop(paste("Invalid sampling method: '", method, "\'", sep=""))
			}
			to_label <- unlabeled[ids$ix[1:batch_size],]
			to_label$score <- ids$x[1:batch_size]
		} else {
			is_active <- FALSE
			warning("Not enough labeled observations in the minority class. Randomly sampling...")
			to_label <- unlabeled[sample(1:nrow(unlabeled), batch_size), ]
		}
		hrule <- "\n——————————————————————————————————————————————"
		for (j in 1:nrow(to_label)) {
			id <- to_label[j,doc_id_col]
			text <- to_label[j,text_col]
			if (method == 'random') {
				message(paste(hrule, "\nDocument ID: ", id, "\n\n", "Document Text: ", "\n\n", text, hrule, sep=''))
			} else {
				score <- to_label[j,"score"]
				message(paste(hrule, "\nDocument ID: ", id, "\n\n", "Score: ", score, "\n\n", "Document Text: ", "\n\n", text, hrule, sep=''))
			}
			while (TRUE) {
				label <- as.integer(readline(prompt=paste("\n\nEnter label for '", y_col, "' (0, 1): ", sep="")))
				if (label %in% c(1,0)) {
					break
				} else {
					warning("Invalid input. You must enter either 0 for false or 1 for true.")
				}
			}
			data[data[doc_id_col] == id, y_col] <- label
			data[data[doc_id_col] == id, 'batch_id'] <- batch_id
			data[data[doc_id_col] == id, 'active'] <- is_active
		}
		if (i < n_batches) {
			message(paste(hrule, "\n\nDone with batch ", batch_id, "/", batch_size, ".\n\nRefitting model with annotations from the previous batch...\n\n", sep=""))
		}
	}
	data
}