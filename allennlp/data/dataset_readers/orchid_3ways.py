from typing import Dict, List, Sequence, Iterable
import itertools
import logging
import random

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

import pickle 

from allennlp.interpret.attackers import adv_utils_thai


logger = logging.getLogger(__name__)


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ""
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":
            return True
        else:
            return False


@DatasetReader.register("orchid_3ways")
class ORCHID3WaysReader(DatasetReader):
    """
    ORCHID NECTEC 
    """
    _VALID_LABELS = { "pos"}
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tag_label: str = "pos",
        feature_labels: Sequence[str] = (),
        label_namespace: str = "labels",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if tag_label is not None and tag_label not in self._VALID_LABELS:
            raise ConfigurationError("unknown tag label type: {}".format(tag_label))
        for label in feature_labels:
            if label not in self._VALID_LABELS:
                raise ConfigurationError("unknown feature label type: {}".format(label))


        self.tag_label = tag_label
        self.feature_labels = set(feature_labels)
        self.label_namespace = label_namespace
        #adv
        self.adv_indexer = SingleIdTokenIndexer(namespace="adversarial_tags", lowercase_tokens=True)
        
    def text_to_instance(self,tokens: List[Token],
                         pos_tags: List[str] = None) -> Instance:
        
        sentence_field = TextField(tokens, self._token_indexers)
        fields = {"text_file": sentence_field}

        if pos_tags:
            pos_field = SequenceLabelField(labels=pos_tags, sequence_field=sentence_field, label_namespace = "pos_tags")
            fields["pos_tags"] = pos_field


        return Instance(fields) 
    
    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        # Create a list to sample contrastsive data
        with open(file_path, "rb") as data_file:
            sentences=pickle.load(data_file)


         
        ############    
        with open(file_path, "rb") as data_file:
            sentences=pickle.load(data_file)
            counter = 0
            idxs = list(range(len(sentences)))


                ###   
            for sentence in sentences:
                fields = [list(field) for field in zip(*sentence)]
                #get neg field
                random_idx  = random.choice(idxs)
                while random_idx == counter:
                    random_idx  = random.choice(idxs)
                neg_fields = sentences[random_idx]
                neg_fields = [list(field) for field in zip(*neg_fields)]
                tokens_, pos_tags = fields
                neg_tokens_, p_tmp = neg_fields
                # TextField requires `Token` objects
                tokens = [Token(token) for token in tokens_]
                neg_tokens = [Token(token) for token in neg_tokens_]
                adv_tokens = []
                adv_tags = []
                counter += 1 
                for token in tokens_:
                    rand_idx = random.randrange(4) # 4 perb type in total
                    perb_fcn = [adv_utils_thai.add_char,
                    adv_utils_thai.del_char,
                    adv_utils_thai.swap_char,
                    adv_utils_thai.replace_char][rand_idx]
                    new_word = random.choice(perb_fcn(token))
                    adv_tokens.append(Token(new_word))
                    if new_word == token:
                        adv_tags.append(0)
                    else:
                        adv_tags.append(rand_idx+1)

                yield self.text_to_instance(tokens,adv_tokens, neg_tokens, pos_tags,adv_tags)    
                
    def text_to_instance(  # type: ignore
        self,
        tokens: List[Token],
        adv_tokens: List[Token],
        neg_tokens: List[Token],
        pos_tags: List[str] = None,
        adv_tags: List[str] = None
    ) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """

        sequence = TextField(tokens, self._token_indexers)
        adv_sequence = TextField(adv_tokens, self._token_indexers)
        neg_sequence = TextField(neg_tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {"tokens": sequence,'adv_tokens':adv_sequence, 'neg_tokens':neg_sequence}
        instance_fields["metadata"] = MetadataField({"words": [x.text for x in tokens], "adv_words": [x.text for x in adv_tokens]
                                                    , "neg_words": [x.text for x in neg_tokens]})
        # Add "feature labels" to instance
        if "pos" in self.feature_labels:
            if pos_tags is None:
                raise ConfigurationError(
                    "Dataset reader was specified to use pos_tags as "
                    "features. Pass them to text_to_instance."
                )
            instance_fields["pos_tags"] = SequenceLabelField(pos_tags, sequence, "pos_tags")


        if self.tag_label == "pos" and pos_tags is not None:
            instance_fields["tags"] = SequenceLabelField(pos_tags, sequence, self.label_namespace)

        #adv 
        if adv_tags:
            instance_fields["adversarial_tags"] = SequenceLabelField(
                adv_tags, sequence, 'adversarial_tags'
            )
        
        return Instance(instance_fields)
