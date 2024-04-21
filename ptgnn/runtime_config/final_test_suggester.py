import typing
from ray.tune.search import Searcher


class FinalSearcher(Searcher):
    """
    Final searcher - Searcher which suggest predefined configurations. Used in the context of being provided the best
    configurations of the parameter search to use the ray framework to execute this configurations in one final test
    at the end - in parallel.
    """
    def __init__(self, best_configs_to_test: typing.Dict[str, dict]):
        """
        Init function of FinalSearcher

        :param best_configs_to_test: best configurations to suggest.
        """
        super(FinalSearcher, self).__init__()
        self.my_configurations = best_configs_to_test
        self.mapping = {}
        self.configurations = {}

    def suggest(self, trial_id: str):
        """
        Suggestion function - suggests the previously set configurations - exactly once each.

        :param trial_id: Trial id of the new trial
        :type trial_id: str
        :return: configuration
        :rtype: dict
        """
        if trial_id in self.configurations:
            return self.configurations[trial_id]
        else:
            mappings = list(self.mapping.values())

            for key, value in self.my_configurations.items():
                if key in mappings:
                    continue
                else:
                    self.mapping[trial_id] = key
                    self.configurations[trial_id] = value
                    return value

            raise Exception("reached end")

    def on_trial_complete(
        self, trial_id: str, result: typing.Optional[typing.Dict] = None, error: bool = False
    ) -> None:
        pass
