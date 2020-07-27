"""
A :class:`~allennlp.data.dataset_readers.dataset_reader.DatasetReader`
reads a file and converts it to a collection of
:class:`~allennlp.data.instance.Instance` s.
The various subclasses know how to read specific filetypes
and produce datasets in the formats required by specific models.
"""


from allennlp.data.dataset_readers.ccgbank import CcgBankDatasetReader
from allennlp.data.dataset_readers.wnut16 import WNUT16DatasetReader
from allennlp.data.dataset_readers.wnut16_adv_2ways import WNUT16Adv2WaysDatasetReader
from allennlp.data.dataset_readers.wnut16_adv_3ways import WNUT16Adv3WaysDatasetReader
from allennlp.data.dataset_readers.conll2003 import Conll2003DatasetReader
from allennlp.data.dataset_readers.conll2003_adv import Conll2003AdvDatasetReader
from allennlp.data.dataset_readers.conll2003_adv_2ways import Conll2003Adv2WaysDatasetReader
from allennlp.data.dataset_readers.conll2003_adv_3ways import Conll2003Adv3WaysDatasetReader
from allennlp.data.dataset_readers.conll2002 import Conll2002DatasetReader
from allennlp.data.dataset_readers.conll2000 import Conll2000DatasetReader
from allennlp.data.dataset_readers.ontonotes_ner import OntonotesNamedEntityRecognition
from allennlp.data.dataset_readers.coreference_resolution import (
    ConllCorefReader,
    PrecoReader,
    WinobiasReader,
)
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.interleaving_dataset_reader import InterleavingDatasetReader
from allennlp.data.dataset_readers.masked_language_modeling import MaskedLanguageModelingReader
from allennlp.data.dataset_readers.next_token_lm import NextTokenLmReader
from allennlp.data.dataset_readers.penn_tree_bank import PennTreeBankConstituencySpanDatasetReader
from allennlp.data.dataset_readers.semantic_role_labeling import SrlReader
from allennlp.data.dataset_readers.semantic_dependency_parsing import (
    SemanticDependenciesDatasetReader,
)
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader
from allennlp.data.dataset_readers.sharded_dataset_reader import ShardedDatasetReader
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import (
    StanfordSentimentTreeBankDatasetReader,
)
from allennlp.data.dataset_readers.quora_paraphrase import QuoraParaphraseDatasetReader
from allennlp.data.dataset_readers.simple_language_modeling import (
    SimpleLanguageModelingDatasetReader,
)
from allennlp.data.dataset_readers.babi import BabiReader
from allennlp.data.dataset_readers.copynet_seq2seq import CopyNetDatasetReader
from allennlp.data.dataset_readers.text_classification_json import TextClassificationJsonReader

from allennlp.data.dataset_readers.orchid import ORCHIDReader
from allennlp.data.dataset_readers.orchid_2ways import ORCHID2WaysReader
from allennlp.data.dataset_readers.orchid_3ways import ORCHID3WaysReader