class DynamicRules:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, pipe_y_sep = 175, score_threshold = 5, upd_value=5) -> None:
        if not hasattr(self, "initialized"):
            self.pipe_y_sep = pipe_y_sep
            self.score_threshold = score_threshold
            self.upd_value = upd_value
            self.min_pipe_y_sep = 175
            self.initialized = True
            self.y_seps = [sep for sep in range(pipe_y_sep, self.min_pipe_y_sep, -upd_value)] + [self.min_pipe_y_sep]
    
    def default_rules(self):
        self.pipe_y_sep = self.min_pipe_y_sep

    def update(self, score_mean):
        if score_mean>self.score_threshold and self.pipe_y_sep>self.min_pipe_y_sep:
            self.pipe_y_sep -= self.upd_value
            self.pipe_y_sep = max(self.min_pipe_y_sep, self.pipe_y_sep)
            print(f"Shorten pipe y separation ({self.pipe_y_sep + self.upd_value} -> {self.pipe_y_sep})")
            return True
        else:
            return False