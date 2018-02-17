class CorpusFeature:
    def __init__(self, name, input_units=None, input_activations=None):
        self._name = name
        self._input_units = input_units
        self._input_activations = input_activations

    @property
    def name(self):
        return self._name

    @property
    def input_units(self):
        return self._input_units

    @property
    def input_activations(self):
        return self._input_activations
