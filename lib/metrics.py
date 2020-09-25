class Metric:
    def __init__(self):
        self.val = 0
        self.N = 0
        self.hist = []

        
    def calc(self, *args):
        raise NotImplementedError
    
    def append(self, *args):
        self.val += self.calc(*args)
        self.N += 1
        
    def result(self):
        return self.val / self.N
    
    def reset(self):
        self.val = 0
        self.N = 0
        
    def history(self):
        self.hist.append(self.val)
    
    
class AccuracyMetric(Metric):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        
    def calc(self, pred, gt):
        return ((pred > self.threshold) == gt.bool()).sum().item() / pred.numel()

# true positive
class LakeAccuracyMetric(Metric):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        
    def calc(self, pred, gt):
        return ((gt == 1) & (pred > self.threshold)).sum().item() / (gt == 1).sum().item()

# false negative    
class NoLakeAccuracyMetric(Metric):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        
    def calc(self, pred, gt):
        return ((gt == 0) & (pred < self.threshold)).sum().item() / (gt == 0).sum().item()

    
class LossMetric(Metric):
    def __init__(self):
        super().__init__()
        #self.threshold = threshold
        
    def calc(self, loss):
        return loss.item()

    

class MIOUMetric(Metric):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        
    def calc(self, pred, gt):
        p1 = (pred > self.threshold)
        p0 = (pred < self.threshold)
        g1 = (g > self.threshold)
        g0 = (g < self.threshold)
        iou1 = (p1 & g1).sum(dim=[1,2,3]) / (p1 | g1).sum(dim=[1,2,3])
        iou0 = (p0 & g0).sum(dim=[1,2,3]) / (p0 | g0).sum(dim=[1,2,3])
        return ((iou1 + iou0) / 2).mean().item()
    
class F1Metric(Metric):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        
    def calc(self, pred, gt):
        p1 = (pred > self.threshold)
        p0 = (pred < self.threshold)
        g1 = (g > self.threshold)
        g0 = (g < self.threshold)
        iou1 = (p1 & g1).sum(dim=[1,2,3]) / pred[0].numel()
        iou0 = (p0 & g0).sum(dim=[1,2,3]) / pred[0].numel()
        return ((iou1 + iou0) / 2).mean().item()