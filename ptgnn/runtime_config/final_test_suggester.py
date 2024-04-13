import typing
from ray.tune.search import Searcher


class FinalSearcher(Searcher):
    def __init__(self, best_configs_to_test: typing.Dict[str, dict]):
        super(FinalSearcher, self).__init__()
        self.my_configurations = best_configs_to_test
        self.mapping = {}
        self.configurations = {}

    def suggest(self, trial_id: str):
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
