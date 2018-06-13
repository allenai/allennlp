from allennlp.data.dataset_readers import DatasetReader

@DatasetReader.register("ccgbank")
class CCGbankDatasetReader(DatasetReader):
