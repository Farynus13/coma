import numpy as np
import onnxruntime as ort
import os
import pandas as pd

from collections import namedtuple
from hashlib import md5
from typing import List, Union, Tuple, Dict

import gym



ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])

class LataccelTokenizer:
  def __init__(self):
    self.vocab_size = VOCAB_SIZE
    self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

  def encode(self, value: Union[float, np.ndarray, List[float]]) -> Union[int, np.ndarray]:
    value = self.clip(value)
    return np.digitize(value, self.bins, right=True)

  def decode(self, token: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    return self.bins[token]

  def clip(self, value: Union[float, np.ndarray, List[float]]) -> Union[float, np.ndarray]:
    return np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])


class TinyPhysicsModel:
  def __init__(self, model_path: str, debug: bool) -> None:
    self.tokenizer = LataccelTokenizer()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    provider = 'CPUExecutionProvider'

    with open(model_path, "rb") as f:
      self.ort_session = ort.InferenceSession(f.read(), options, [provider])

  def softmax(self, x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

  def predict(self, input_data: dict, temperature=1.) -> int:
    res = self.ort_session.run(None, input_data)[0]
    probs = self.softmax(res / temperature, axis=-1)
    # we only care about the last timestep (batch size is just 1)
    assert probs.shape[0] == 1
    assert probs.shape[2] == VOCAB_SIZE
    sample = np.random.choice(probs.shape[2], p=probs[0, -1])
    return sample

  def get_current_lataccel(self, sim_states: List[State], actions: List[float], past_preds: List[float]) -> float:
    tokenized_actions = self.tokenizer.encode(past_preds)
    raw_states = [list(x) for x in sim_states]
    states = np.column_stack([actions, raw_states])
    input_data = {
      'states': np.expand_dims(states, axis=0).astype(np.float32),
      'tokens': np.expand_dims(tokenized_actions, axis=0).astype(np.int64)
    }
    return self.tokenizer.decode(self.predict(input_data, temperature=0.8))

FUTURE_PLAN_STEPS = FPS * 5  # 5 secs
class TinyPhysicsSimulator:
  def __init__(self, model: TinyPhysicsModel, data_path: str) -> None:
    self.data_path = data_path
    self.sim_model = model
    self.data = self.get_data(data_path)
    self.reset()

  def reset(self) -> None:
    self.step_idx = CONTEXT_LENGTH
    state_target_futureplans = [self.get_state_target_futureplan(i) for i in range(self.step_idx)]
    self.state_history = [x[0] for x in state_target_futureplans]
    self.action_history = self.data['steer_command'].values[:self.step_idx].tolist()
    self.current_lataccel_history = [x[1] for x in state_target_futureplans]
    self.target_lataccel_history = [x[1] for x in state_target_futureplans]
    self.target_future = None
    self.current_lataccel = self.current_lataccel_history[-1]
    seed = int(md5(self.data_path.encode()).hexdigest(), 16) % 10**4
    np.random.seed(seed)

  def get_data(self, data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    processed_df = pd.DataFrame({
      'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
      'v_ego': df['vEgo'].values,
      'a_ego': df['aEgo'].values,
      'target_lataccel': df['targetLateralAcceleration'].values,
      'steer_command': -df['steerCommand'].values  # steer commands are logged with left-positive convention but this simulator uses right-positive
    })
    return processed_df

  def sim_step(self, step_idx: int) -> None:
    pred = self.sim_model.get_current_lataccel(
      sim_states=self.state_history[-CONTEXT_LENGTH:],
      actions=self.action_history[-CONTEXT_LENGTH:],
      past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:]
    )
    pred = np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
    if step_idx >= CONTROL_START_IDX:
      self.current_lataccel = pred
    else:
      self.current_lataccel = self.get_state_target_futureplan(step_idx)[1]

    self.current_lataccel_history.append(self.current_lataccel)

  def get_state_target_futureplan(self, step_idx: int) -> Tuple[State, float, FuturePlan]:
    state = self.data.iloc[step_idx]
    return (
      State(roll_lataccel=state['roll_lataccel'], v_ego=state['v_ego'], a_ego=state['a_ego']),
      state['target_lataccel'],
      FuturePlan(
        lataccel=self.data['target_lataccel'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
        roll_lataccel=self.data['roll_lataccel'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
        v_ego=self.data['v_ego'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
        a_ego=self.data['a_ego'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist()
      )
    )

  def step_init(self) -> None:
    state, target, futureplan = self.get_state_target_futureplan(self.step_idx)
    self.state_history.append(state)
    self.target_lataccel_history.append(target)
    self.futureplan = futureplan
    return self.target_lataccel_history[self.step_idx], self.current_lataccel, self.state_history[self.step_idx], self.futureplan

  def step_action(self,action) -> None:

    if self.step_idx < CONTROL_START_IDX:
      action = self.data['steer_command'].values[self.step_idx]
    action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
    self.action_history.append(action)
    
    self.sim_step(self.step_idx)
    self.step_idx += 1

  def compute_cost(self) -> Dict[str, float]:
    target = np.array(self.target_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
    pred = np.array(self.current_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]

    lat_accel_cost = np.mean((target - pred)**2) * 100
    jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
    total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
    return {'lataccel_cost': lat_accel_cost, 'jerk_cost': jerk_cost, 'total_cost': total_cost}
  
  def get_reward(self) -> float:
    i = len(self.current_lataccel_history) - 1
    target = np.array(self.target_lataccel_history)[i]
    pred = np.array(self.current_lataccel_history)[-2:]

    lat_accel_cost = np.mean((target - pred[-1])**2)
    jerk_cost = np.mean((np.diff(pred) / DEL_T)**2)
    total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
    return -total_cost

class Comma(gym.Env):

  def __init__(self, task, id, size=(64, 64),seed=42):
    self._size = size
    self.model_path = './models/tinyphysics.onnx'
    self.data_dir = './data'
    #all csv files in data_dir
    self.data_list = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]

  
  def reset(self):
    data_path = self.data_dir + '/' + np.random.choice(self.data_list)
    self.tinyphysicsmodel = TinyPhysicsModel(self.model_path, debug=False)
    self.sim = TinyPhysicsSimulator(self.tinyphysicsmodel, str(data_path))
    return self._obs(is_first=True)

  @property
  def observation_space(self):
      
      return gym.spaces.Dict(
          {
              # "image": gym.spaces.Box(0, 255, self._map_size, dtype=np.uint8),
              "is_first": gym.spaces.Box(0,1, shape=(1,), dtype=np.uint8),
              "is_last": gym.spaces.Box(0,1, shape=(1,), dtype=np.uint8),
              "is_terminal": gym.spaces.Box(0,1, shape=(1,), dtype=np.uint8),
              "lataccel": gym.spaces.Box(-np.inf, np.inf, shape=(2,),dtype=np.float32),#"log_player_pos": gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32
              "state": gym.spaces.Box(-np.inf, np.inf, shape=(3,),dtype=np.float32),
              "future_plan": gym.spaces.Box(-np.inf, np.inf, shape=(4*(FUTURE_PLAN_STEPS-1),),dtype=np.float32),
              "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),

          }
      )

  @property
  def action_space(self):
    return gym.spaces.Box(-2.0, 2.0, (1,), dtype=np.float32)
  
  def _obs(self, is_first=False, is_last=False, is_terminal=False):
    target_lataccel,current_lataccel,state_history,futureplan = self.sim.step_init()
    state = np.array([state_history.roll_lataccel, state_history.v_ego, state_history.a_ego])
    fp = []
    for i in range(FUTURE_PLAN_STEPS-1):
      fp.append(futureplan.lataccel[i])
      fp.append(futureplan.roll_lataccel[i])
      fp.append(futureplan.v_ego[i])
      fp.append(futureplan.a_ego[i])

    lataccel = np.array([target_lataccel,current_lataccel])
      
    
    fp = np.array(fp)
    d = dict(
        is_first=bool(is_first),
        is_last=bool(is_last),
        is_terminal=bool(is_terminal),
        lataccel=np.float32(lataccel),
        state=np.float32(state),
        future_plan=np.float32(fp),
        image=np.zeros(self._size + (3,), dtype=np.uint8),
    )
    # print(d)
    return d
  def step(self, action):
    self.sim.step_action(action)

    reward = self.sim.get_reward() 
    self._done = 0 if self.sim.step_idx < COST_END_IDX else 1
    
    return self._obs(is_last=self._done, is_terminal=self._done), np.float32(reward),self._done, {}



