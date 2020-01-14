from allennlp.interpret.attackers.attacker import Attacker
from allennlp.predictors import Predictor


@Attacker.register("blackbox")
class BlackBox(Attacker):
    """
    """

    def __init__(self, predictor: Predictor) -> None:
        super().__init__(predictor)
