import torch

class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_bg,self.next_edge,_, _, _,_ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_bg = None
            self.next_edge = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_bg = self.next_bg.cuda(non_blocking = True)
            self.next_edge = self.next_edge.cuda(non_blocking = True)
            
            self.next_input = self.next_input.float() #if need
            self.next_target = self.next_target.float() #if need
            self.next_bg = self.next_bg.float() #if need
            self.next_edge = self.next_edge.float() #if need

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        bg     = self.next_bg
        edge     = self.next_edge
        self.preload()
        return input, target, bg,edge