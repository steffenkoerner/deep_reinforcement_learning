
class Config:
  def __init__(self):
    self.replay_buffer_size = int(1e5)
    self.batch_size = 64
    self.gamma = 0.99
    self.tau = 1e-3
    self.learning_rate = 5e-4
