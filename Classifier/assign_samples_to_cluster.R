library("caret")
neuralnet_fit_loaded=readRDS("D:/OneDrive/post-doc/colaboracions/clusters clinics/Gens extesos/neuralnet_fit.rds") #Load trained neural net 
sig_features=c("TGFB1","ZEB1","FAP","ZEB2","GLUL","ENTPD1","GOT1","LDHA","TWIST1","GLS") #List of features of the signature
gene_name_column="Gene.name" #Column with gene id
gene_data_file="D:/OneDrive/Colon/Busca de signatures/TCGA/vst_gene_expression/TCGA_MSS_COAD_gene_expression_vst_mod.csv" #File with gene expression. Must only have a column with gene names followed by columns with the normalyzed log like gene expression for each patient

gene_data=read.csv(gene_data_file,header = T) #Read gene data
#Select only signature genes
n_sigfeatures=which(gene_data[,gene_name_column] %in% sig_features)
gene_data_subset=gene_data[n_sigfeatures,]
rownames(gene_data_subset)=gene_data_subset[,gene_name_column]
gene_data_subset=gene_data_subset[sig_features,]
gene_data_subset=gene_data_subset[,which(colnames(gene_data_subset)!=gene_name_column)]
#Scale data to mean of 0 and SD of 1
scaled_gene_data_subset_t=scale(t(gene_data_subset),center=TRUE,scale=TRUE)
#Predict cluster
prediction=predict(neuralnet_fit_loaded, newdata = scaled_gene_data_subset_t)
prediction_prob=predict(neuralnet_fit_loaded, newdata = scaled_gene_data_subset_t,type="prob") #Get probability

#Create outputs
prediction_data_frame=data.frame(cbind(prediction,prediction_prob),row.names =row.names(scaled_gene_data_subset_t) )
write.csv(prediction_data_frame,"prediction_data_frame.csv")
freq_table=prop.table(table(prediction))
write.csv(freq_table,"freq_table.csv")

