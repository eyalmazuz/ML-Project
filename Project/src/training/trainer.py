

class Trainer:
    
    def __init__(self, model, X, y, cv, multi_class, **kwargs):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv
        self.multi_class = multi_class
        self.kwargs = kwargs

    def train(self,):
        pass

