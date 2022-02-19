
class Config:
  def __init__(self):
    self.replay_buffer_size = int(1e5)
    self.batch_size = 64
    self.gamma = 0.99
    self.tau = 1e-3
    self.learning_rate = 5e-4
    
    self.number_episodes = 2000
    self.eps_start = 1.0
    self.eps_end = 0.01
    self.eps_decay = 0.995

    self.stop_return = 13
    self.save_each_return_step = 3

    #TODO Move config for Neural Network to here
    self.input_layer_size = 37
    self.output_layer_size = 4
