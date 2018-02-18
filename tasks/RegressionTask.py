class RegressionTask:
    def __init__(self, name, output_units=None, output_activations=None, weight=1, generate_method=None,
                 normalize=lambda label: RegressionTask.default_normalization(label),
                 normalize_inflow=False
                 ):
        self._name = name
        self._output_units = output_units
        self._output_activations = output_activations
        self._weight = weight
        self._generate_method = generate_method
        self._normalization = normalize
        self._inflow_normalization = normalize_inflow

    @staticmethod
    def default_normalization(label):
        min_value = label.min()
        max_value = label.max()
        ratio = 0  # 0 - 1
        # ratio = 1/2  # 0.25 - 0.75
        # ratio = 1  # 0 - 1
        # ratio = 3/2 #0.33 - 0.66
        min2 = min_value - ratio * (max_value - min_value)
        max2 = max_value + ratio * (max_value - min_value)
        return (label - min2) / (max2 - min2)

    @property
    def name(self):
        return self._name

    @property
    def output_units(self):
        return self._output_units

    @property
    def output_activations(self):
        return self._output_activations

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    @property
    def generate_method(self):
        return self._generate_method

    @property
    def normalization(self):
        return self._normalization

    @property
    def inflow_normalization(self):
        return self._inflow_normalization
