class ClassificationTask:
    def __init__(self, name, output_units=None, output_activations=None, weight=1, nb_classes=2, generate_method=None):
        self._name = name
        self._output_units = output_units
        self._output_activations = output_activations
        self._weight = weight
        self._generate_method = generate_method
        self._nb_classes = nb_classes

    @property
    def name(self):
        return self._name

    @property
    def output_units(self):
        return self._output_units

    @property
    def output_activation(self):
        return self._output_activations

    @property
    def weight(self):
        return self._weight

    @property
    def nb_classes(self):
        return self._nb_classes

    @property
    def generate_method(self):
        return self._generate_method
