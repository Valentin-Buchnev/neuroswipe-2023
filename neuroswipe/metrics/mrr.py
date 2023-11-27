class MRR:
    def __init__(self):
        self.reset()

    def reset(self):
        self.top = [0.0, 0.0, 0.0, 0.0]
        self.count = 0.0

    def update(self, pred, gt):
        if pred[0] == gt:
            self.top[0] += 1
        elif pred[1] == gt:
            self.top[1] += 1
        elif pred[2] == gt:
            self.top[2] += 1
        elif pred[3] == gt:
            self.top[3] += 1
        self.count += 1

    def compute(self):
        return (self.top[0] + self.top[1] * 0.1 + self.top[2] * 0.09 + self.top[3] * 0.08) / self.count
