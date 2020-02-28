from copy import copy, deepcopy
from typing import Dict, List, Tuple

import numpy
import torch
import random

from allennlp.common.util import JsonDict, sanitize
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import (
    ELMoTokenCharactersIndexer,
    TokenCharactersIndexer,
    SingleIdTokenIndexer,
)
from allennlp.data.tokenizers import Token
from allennlp.interpret.attackers import utils
from allennlp.interpret.attackers.attacker import Attacker
from allennlp.data import Instance

from allennlp.modules.token_embedders import Embedding

from allennlp.nn import util
from allennlp.predictors.predictor import Predictor

DEFAULT_IGNORE_TOKENS = ["@@NULL@@", ".", ",", ";", "!", "?", "[MASK]", "[SEP]", "[CLS]"]
CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
# note :
# hotflip isalnum is used to exclude alphnumeric string
@Attacker.register("blackbox")
class BlackBox(Attacker):
    """
    """

    def __init__(self, predictor: Predictor, vocab_namespace: str = "tokens", max_tokens: int = 5000) -> None:
        super().__init__(predictor)
        self.vocab = self.predictor._model.vocab
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

    def _make_embedder_input(self, all_tokens: List[str]) -> Dict[str, torch.Tensor]:
        inputs = {}
        # A bit of a hack; this will only work with some dataset readers, but it'll do for now.
        indexers = self.predictor._dataset_reader._token_indexers  # type: ignore
        for indexer_name, token_indexer in indexers.items():
            if isinstance(token_indexer, SingleIdTokenIndexer):
                all_indices = [
                    self.vocab.get_token_index(token, self.namespace) for token in all_tokens
                ]
                inputs[indexer_name] = {"tokens": torch.LongTensor(all_indices).unsqueeze(0)}
            elif isinstance(token_indexer, TokenCharactersIndexer):
                tokens = [Token(x) for x in all_tokens]
                max_token_length = max(len(x) for x in all_tokens)
                # sometime max_token_length is too short for cnn encoder
                max_token_length = max(max_token_length, token_indexer._min_padding_length)
                indexed_tokens = token_indexer.tokens_to_indices(tokens, self.vocab)
                padding_lengths = token_indexer.get_padding_lengths(indexed_tokens)
                padded_tokens = token_indexer.as_padded_tensor_dict(indexed_tokens, padding_lengths)
                inputs[indexer_name] = {
                    "token_characters": torch.LongTensor(
                        padded_tokens["token_characters"]
                    ).unsqueeze(0)
                }
            elif isinstance(token_indexer, ELMoTokenCharactersIndexer):
                elmo_tokens = []
                for token in all_tokens:
                    elmo_indexed_token = token_indexer.tokens_to_indices(
                        [Token(text=token)], self.vocab, "sentence"
                    )["sentence"]
                    elmo_tokens.append(elmo_indexed_token[0])
                inputs[indexer_name] = {"tokens": torch.LongTensor(elmo_tokens).unsqueeze(0)}
            else:
                raise RuntimeError("Unsupported token indexer:", token_indexer)    

        return util.move_to_device(inputs, self.cuda_device)
                
    def attack_from_instance(
        self,
        input_instance: Instance,
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

        Parameters
        ----------
        input_instance: ``Instance``,
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
            return sanitize({"final": [original_tokens], "original": original_tokens})

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

        # Because we can save computation by getting grads and outputs at the same time, we do
        # them together at the end of the loop, even though we use grads at the beginning and
        # outputs at the end.  This is our initial gradient for the beginning of the loop.  The
        # output can be ignored here.
        grads, outputs = self.predictor.get_gradients([instance])

        # Ignore any token that is in the ignore_tokens list by setting the token to already
        # flipped.
        flipped: List[int] = []
        attacked:  List[int] = []
        for index, token in enumerate(text_field.tokens):
            if token.text in ignore_tokens or len(token.text) <= 3:
                flipped.append(index)
            else: 
                # Get new token using taylor approximation.
                new_word = random.choice(self.add_char(token.text))
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
        #adversarial_list
        adv_list = []       
        for i in range(len(text_field.tokens)):
            if i in attacked:
                adv_list.append("adversarial")
            else:
                adv_list.append("clean")
                
#         input_instance.add_field(
#             "adv_category", SequenceLabelField(output_dict["adv_category"], text_field), self.predictor._model.vocab
#         )
    
#         grads, outputs = self.predictor.get_gradients([instance])  # predictions

#         for key, output in outputs.items():
#             if isinstance(output, torch.Tensor):
#                 outputs[key] = output.detach().cpu().numpy().squeeze()
#             elif isinstance(output, list):
#                 outputs[key] = output[0]

#             # TODO(mattg): taking the first result here seems brittle, if we're in a case where
#             # there are multiple predictions.
#         adv_labeled_instances = self.predictor.predictions_to_labeled_instances(
#             instance, outputs
#         )
            

        final_tokens.append(text_field.tokens)
        #no output version for training only
        return sanitize({"final": final_tokens, "original": original_tokens})

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
    
    def add_char(self, w_original):
        """
        add letter in the middle of a word
        """
        word_list = []
        if len(w_original) > 3:
            for i in range(1, len(w_original)):
                for char in CHARACTERS:
                    w = copy(w_original)
                    w = list(w)
                    w[i] = char + w[i]
                    word_list.append("".join(w))
            return word_list
        else:
            return [w_original]