from . import BaseController
from our_dreamer.dreamer_model import get_model
import numpy as np


class Controller(BaseController):
  """
  An rl controller
  """
  def __init__(self):
    model_path = 'logdir/comma-controls-1/latest.pt'
    self.model = get_model(path=model_path)
    self.agent_state = None # no initial state
    self.is_first = True


  def update(self, target_lataccel, current_lataccel, state, future_plan):
    state_array = np.array([state.roll_lataccel, state.v_ego, state.a_ego])
    fp = []
    short_of = 49 - len(future_plan.lataccel)
    for i in range(len(future_plan.lataccel)):
      fp.append(future_plan.lataccel[i])
      fp.append(future_plan.roll_lataccel[i])
      fp.append(future_plan.v_ego[i])
      fp.append(future_plan.a_ego[i])
    for i in range(short_of):
      fp.append(0)
      fp.append(0)
      fp.append(0)
      fp.append(0)

    lataccel = np.array([target_lataccel,current_lataccel])
      
    
    fp = np.array(fp)
    obs = dict(
        is_first=bool(self.is_first),
        is_last=bool(False),
        is_terminal=bool(False),
        lataccel=np.float32(lataccel),
        state=np.float32(state_array),
        future_plan=np.float32(fp),
        image=np.zeros((64,64) + (3,), dtype=np.uint8),
    )
    obs = [obs]
    obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}

    self.is_first = False

    action, self.agent_state = self.model(obs, self.agent_state)
    action = np.array(action['action'][0].detach().cpu())  
    action = action * 2.0      

    return action
