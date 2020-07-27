from copy import copy, deepcopy
from typing import Dict, List, Tuple

import numpy
import torch
import random
from allennlp.interpret.attackers import adv_utils
from allennlp.common.util import JsonDict, sanitize
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import (
    ELMoTokenCharactersIndexer,
    TokenCharactersIndexer,
    SingleIdTokenIndexer,
)
from allennlp.data.tokenizers import Token
from allennlp.interpret.attackers import utils, adv_utils_thai
from allennlp.interpret.attackers.attacker import Attacker
from allennlp.data import Instance

from allennlp.modules.token_embedders import Embedding

from allennlp.nn import util
from allennlp.predictors.predictor import Predictor

DEFAULT_IGNORE_TOKENS = ["@@NULL@@", ".", ",", ";", "!", "?", "[MASK]", "[SEP]", "[CLS]"]
consonant="กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮ"
front_vowel="เแโใไ"
lower_vowel="อุอู".replace('อ','') #ทำแบบนี้จะได้อ่านออก
rear_vowel = "าําๅๆะฯๅๆ"
upper_vowel = "อ็อ้อ์อิอีอือึอํอัอ่อ๋อ๊".replace('อ','')
tone = "อ้อ่อ๋อ๊".replace('อ','')

CHARACTERS = consonant+front_vowel+lower_vowel+rear_vowel+upper_vowel+tone
# note :
# hotflip isalnum is used to exclude alphnumeric string
@Attacker.register("thaibox")
class ThaiBox(Attacker):
    """
    """

    def __init__(self, predictor: Predictor, candidates: dict, vocab_namespace: str = "tokens", max_tokens: int = 5000) -> None:
        super().__init__(predictor)
        self.vocab = self.predictor._model.vocab
        self.candidates = candidates
        self.namespace = vocab_namespace
        # Force new tokens to be alphanumeric
        self.max_tokens = max_tokens
        self.invalid_replacement_indices: List[int] = []
        for i in self.vocab._index_to_token[self.namespace]:
            if not self.vocab._index_to_token[self.namespace][i].isalnum():
                self.invalid_replacement_indices.append(i)
        self.embedding_matrix: torch.Tensor = None
        self.embedding_layer: torch.nn.Module = None
        # get device number
        self.cuda_device = predictor.cuda_device


    def attack_from_instance(
        self,
        input_instance: Instance,
        input_field_to_attack: str = "tokens",
        grad_input_field: str = "grad_input_1",
        ignore_tokens: List[str] = None,
        target: JsonDict = None,
    ) -> JsonDict:
        """
        """
        if self.embedding_matrix is None:
            self.initialize()
        ignore_tokens = DEFAULT_IGNORE_TOKENS if ignore_tokens is None else ignore_tokens

        # If `target` is `None`, we move away from the current prediction, otherwise we move
        # _towards_ the target.
        sign = -1 if target is None else 1

        if target is None:
            output_dict = self.predictor._model.forward_on_instance(input_instance)
        else:
            output_dict = target

        # This now holds the predictions that we want to change (either away from or towards,
        # depending on whether `target` was passed).  We'll use this in the loop below to check for
        # when we've met our stopping criterion.

        #         original_instances = self.predictor.predictions_to_labeled_instances(input_instance, output_dict)
        # This is just for ease of access in the UI, so we know the original tokens.  It's not used
        # in the logic below.

        ###my mod: all tags###
        text_field: TextField = input_instance["tokens"]
        input_instance.add_field(
            "tags", SequenceLabelField(output_dict["tags"], text_field), self.predictor._model.vocab
        )
        original_text_field: TextField = input_instance[  # type: ignore
            input_field_to_attack
        ]
        ###my mod: all tags###
        #         original_text_field: TextField = original_instances[0][  # type: ignore
        #             input_field_to_attack
        #         ]
        original_tokens = deepcopy(original_text_field.tokens)

        # if all tags are Os, don't attack.
        if all(tag == "O" for tag in output_dict["tags"]):
            return sanitize({"final": [original_tokens], "original": original_tokens
                             ,'adversarial_list':['clean']*len(original_tokens)})

        final_tokens = []
        # `original_instances` is a list because there might be several different predictions that
        # we're trying to attack (e.g., all of the NER tags for an input sentence).  We attack them
        # one at a time.
        #         instance = random.choice(original_instances)

        instance = deepcopy(input_instance)

        # Gets a list of the fields that we want to check to see if they change.
        fields_to_compare = utils.get_fields_to_compare_instance(
            input_instance, instance, input_field_to_attack
        )

        # We'll be modifying the tokens in this text field below, and grabbing the modified
        # list after the `while` loop.
        text_field: TextField = instance[input_field_to_attack]  # type: ignore

        # Ignore any token that is in the ignore_tokens list by setting the token to already
        # flipped.
        flipped: List[int] = []
        attacked:  List[int] = []
        attacked_type:  List[int] = []
        for index, token in enumerate(text_field.tokens):
            if token.text in ignore_tokens or len(token.text) <= 3:
                flipped.append(index)
            else: 
                # Get new token using taylor approximation.
#                 new_word = random.choice(self.candidates[token.text])
                perb_num = random.randrange(4) # 4 perb type in total
                perb_fcn = [adv_utils_thai.add_char,
                            adv_utils_thai.del_char,
                            adv_utils_thai.swap_char,
                            adv_utils_thai.replace_char][perb_num]
                new_word = random.choice(perb_fcn(token.text))
                
                # Flip token.  We need to tell the instance to re-index itself, so the text field
                # will actually update.
                new_token = Token(
                    new_word
                    #                     self.vocab._index_to_token[self.namespace][new_id]
                )  # type: ignore
                text_field.tokens[index] = new_token
                instance.indexed = False
                flipped.append(index)
                attacked.append(index)
                attacked_type.append(perb_num)
        #adversarial_list
        adv_list = []       
        for i in range(len(text_field.tokens)):
            if i in attacked:
                adv_list.append(attacked_type[attacked.index(i)])
            else:
                adv_list.append("clean")
                


        final_tokens.append(text_field.tokens)
        #no output version for training only
        return sanitize({"final": final_tokens, "original": original_tokens,"adversarial_list":adv_list})

#         return sanitize({"final": final_tokens, "original": original_tokens, "outputs": outputs})

    def attack_from_json(
        self,
        inputs: JsonDict,
        input_field_to_attack: str = "tokens",
        grad_input_field: str = "grad_input_1",
        ignore_tokens: List[str] = None,
        target: JsonDict = None,
    ) -> JsonDict:
        """
        Replaces one token at a time from the input until the model's prediction changes.
        ``input_field_to_attack`` is for example ``tokens``, it says what the input field is
        called.  ``grad_input_field`` is for example ``grad_input_1``, which is a key into a grads
        dictionary.

        The method computes the gradient w.r.t. the tokens, finds the token with the maximum
        gradient (by L2 norm), and replaces it with another token based on the first-order Taylor
        approximation of the loss.  This process is iteratively repeated until the prediction
        changes.  Once a token is replaced, it is not flipped again.

        # Parameters

        inputs : ``JsonDict``
            The model inputs, the same as what is passed to a ``Predictor``.
        input_field_to_attack : ``str``, optional (default='tokens')
            The field that has the tokens that we're going to be flipping.  This must be a
            ``TextField``.
        grad_input_field : ``str``, optional (default='grad_input_1')
            If there is more than one field that gets embedded in your model (e.g., a question and
            a passage, or a premise and a hypothesis), this tells us the key to use to get the
            correct gradients.  This selects from the output of :func:`Predictor.get_gradients`.
        ignore_tokens : ``List[str]``, optional (default=DEFAULT_IGNORE_TOKENS)
            These tokens will not be flipped.  The default list includes some simple punctuation,
            OOV and padding tokens, and common control tokens for BERT, etc.
        target : ``JsonDict``, optional (default=None)
            If given, this will be a `targeted` hotflip attack, where instead of just trying to
            change a model's prediction from what it current is predicting, we try to change it to
            a `specific` target value.  This is a ``JsonDict`` because it needs to specify the
            field name and target value.  For example, for a masked LM, this would be something
            like ``{"words": ["she"]}``, because ``"words"`` is the field name, there is one mask
            token (hence the list of length one), and we want to change the prediction from
            whatever it was to ``"she"``.
        """
        #         if self.embedding_matrix is None:
        #             self.initialize()
        #         ignore_tokens = DEFAULT_IGNORE_TOKENS if ignore_tokens is None else ignore_tokens

        #         # If `target` is `None`, we move away from the current prediction, otherwise we move
        #         # _towards_ the target.
        #         sign = -1 if target is None else 1
        instance = self.predictor._json_to_instance(inputs)
        return self.attack_from_instance(instance)
    
#     def add_char(self, w_original):
#         """
#         add letter in the middle of a word
#         """
#         word_list = []
#         if len(w_original) > 3:
#             for i in range(1, len(w_original)):
#                 for char in CHARACTERS:
#                     w = copy(w_original)
#                     w = list(w)
#                     w[i] = char + w[i]
#                     word_list.append("".join(w))
#             return word_list
#         else:
#             return [w_original]