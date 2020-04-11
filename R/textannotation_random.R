#' Text Annotation Using Random Sampling
#'
#' This function supports annotation of documents using random sampling.
#' @param data A dataframe with a 'text' column and 'doc_id' column.
#' @param text_col A string with the name of the column containing document text. Defaults to 'text'
#' @param y_col A string with the name of the column containing document text. Defaults to 'y'
#' @param doc_id_col A string with the name of the column containing document ids. Defaults to 'doc_id'
#' @param n_docs An integer for the number of documents to label in the session.
#' @return An data frame object derived from the data frame specified in
#' the 'data' parameter where the new annotations have been added to the
#' the column specified in 'y_col', a new column 'active' has been added
#' to specifying whether or not a newly-labeled document was sampled 
#' actively or passively (randomly).
#' @author Blake Miller
#' @keywords active learning text supervised learning annotation
#' @export
#' @examples
#' data <- read.csv('hate_speech.csv', stringsAsFactors=FALSE)
#' data_annotated <- textannotation_random(data, text_col='comment_text', y_col='toxic', doc_id_col='id', n_docs=100)

textannotation_random <- function(data, text_col='text', y_col='y', doc_id_col='doc_id', n_docs=100) {
	if (!y_col %in% names(data)) {
		warning(paste("No existing y column found in corpus data frame. Creating new column: '", y_col, "\'", sep=""))
		data[y_col] <- NA
	}
	if (!'active' %in% names(data)) {
		data$active <- NA
	}
	unlabeled <- data[which(is.na(data[y_col])),]
	to_label <- unlabeled[sample(1:nrow(unlabeled), n_docs), ]
	hrule <- "\n——————————————————————————————————————————————"
	for (i in 1:nrow(to_label)) {
		id <- to_label[i,doc_id_col]
		text <- to_label[i,text_col]
		message(paste(hrule, "\nDocument ID: ", id, "\n\n", "Document Text: ", "\n\n", text, hrule, sep=''))
		while (TRUE) {
			label <- as.integer(readline(prompt=paste("\n\nEnter label for '", y_col, "' (0, 1): ", sep="")))
			if (label %in% c(1,0)) {
				break
			} else {
				warning("Invalid input. You must enter either 0 for false or 1 for true.")
			}
		}
		data[data[doc_id_col] == id, y_col] <- label
		data[data[doc_id_col] == id, 'active'] <- FALSE
	}
	data
}