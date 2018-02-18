class CorpusFeature:

    def __init__(self, name, type='float', vocabulary=None, embedding_units=None, input_units=None, input_activations=None,
                 normalize=lambda feature: CorpusFeature.default_normalization(feature),
                 normalize_inflow=False
                 ):
        self._name = name
        self._type = type
        self._vocabulary = vocabulary
        self._embedding_units = embedding_units
        self._input_units = input_units
        self._input_activations = input_activations
        self._normalization = normalize
        self._inflow_normalization = normalize_inflow

    @staticmethod
    def default_normalization(feature):
        min_value = feature.min()
        max_value = feature.max()
        ratio = 0  # 0 - 1
        # ratio = 1/2  # 0.25 - 0.75
        # ratio = 1  # 0 - 1
        # ratio = 3/2 #0.33 - 0.66
        min2 = min_value - ratio * (max_value - min_value)
        max2 = max_value + ratio * (max_value - min_value)
        return (feature - min2) / (max2 - min2)

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def vocabulary(self):
        return self._vocabulary

    @property
    def embedding_units(self):
        return self._embedding_units

    @property
    def input_units(self):
        return self._input_units

    @property
    def input_activations(self):
        return self._input_activations

    @property
    def normalization(self):
        return self._normalization

    @property
    def inflow_normalization(self):
        return self._inflow_normalization
