import argparse
import os
import pathlib
import sys
import gym
import ruamel.yaml as yaml
from torch import distributions as torchd

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent))

import tools

import torch
from torch import nn
from networks import MultiEncoder,RSSM,MultiDecoder,MLP
import copy

to_np = lambda x: x.detach().cpu().numpy()


class Random(nn.Module):
    def __init__(self, config, act_space):
        super(Random, self).__init__()
        self._config = config
        self._act_space = act_space

    def actor(self, feat):
        if self._config.actor["dist"] == "onehot":
            return tools.OneHotDist(
                torch.zeros(
                    self._config.num_actions, device=self._config.device
                ).repeat(self._config.envs, 1)
            )
        else:
            return torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(
                        self._act_space.low, device=self._config.device
                    ).repeat(self._config.envs, 1),
                    torch.tensor(
                        self._act_space.high, device=self._config.device
                    ).repeat(self._config.envs, 1),
                ),
                1,
            )

    def train(self, start, context, data):
        return None, {}


class Plan2Explore(nn.Module):
    def __init__(self, config, world_model, reward):
        super(Plan2Explore, self).__init__()
        self._config = config
        self._use_amp = True if config.precision == 16 else False
        self._reward = reward
        self._behavior = ImagBehavior(config, world_model)
        self.actor = self._behavior.actor
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
            stoch = config.dyn_stoch * config.dyn_discrete
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
            stoch = config.dyn_stoch
        size = {
            "embed": world_model.embed_size,
            "stoch": stoch,
            "deter": config.dyn_deter,
            "feat": config.dyn_stoch + config.dyn_deter,
        }[self._config.disag_target]
        kw = dict(
            inp_dim=feat_size
            + (
                config.num_actions if config.disag_action_cond else 0
            ),  # pytorch version
            shape=size,
            layers=config.disag_layers,
            units=config.disag_units,
            act=config.act,
        )
        self._networks = nn.ModuleList(
            [MLP(**kw) for _ in range(config.disag_models)]
        )
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._expl_opt = tools.Optimizer(
            "explorer",
            self._networks.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            **kw
        )

    def train(self, start, context, data):
        with tools.RequiresGrad(self._networks):
            metrics = {}
            stoch = start["stoch"]
            if self._config.dyn_discrete:
                stoch = torch.reshape(
                    stoch, (stoch.shape[:-2] + ((stoch.shape[-2] * stoch.shape[-1]),))
                )
            target = {
                "embed": context["embed"],
                "stoch": stoch,
                "deter": start["deter"],
                "feat": context["feat"],
            }[self._config.disag_target]
            inputs = context["feat"]
            if self._config.disag_action_cond:
                inputs = torch.concat(
                    [inputs, torch.tensor(data["action"], device=self._config.device)],
                    -1,
                )
            metrics.update(self._train_ensemble(inputs, target))
        metrics.update(self._behavior._train(start, self._intrinsic_reward)[-1])
        return None, metrics

    def _intrinsic_reward(self, feat, state, action):
        inputs = feat
        if self._config.disag_action_cond:
            inputs = torch.concat([inputs, action], -1)
        preds = torch.cat(
            [head(inputs, torch.float32).mode()[None] for head in self._networks], 0
        )
        disag = torch.mean(torch.std(preds, 0), -1)[..., None]
        if self._config.disag_log:
            disag = torch.log(disag)
        reward = self._config.expl_intr_scale * disag
        if self._config.expl_extr_scale:
            reward += self._config.expl_extr_scale * self._reward(feat, state, action)
        return reward

    def _train_ensemble(self, inputs, targets):
        with torch.cuda.amp.autocast(self._use_amp):
            if self._config.disag_offset:
                targets = targets[:, self._config.disag_offset :]
                inputs = inputs[:, : -self._config.disag_offset]
            targets = targets.detach()
            inputs = inputs.detach()
            preds = [head(inputs) for head in self._networks]
            likes = torch.cat(
                [torch.mean(pred.log_prob(targets))[None] for pred in preds], 0
            )
            loss = -torch.mean(likes)
        metrics = self._expl_opt(loss, self._networks.parameters())
        return metrics


class WorldModel(nn.Module):
    def __init__(self, obs_space, config):
        super(WorldModel, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )
    
    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)
    
class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()

class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

class Dreamer(nn.Module):
    def __init__(self, obs_space,act_space, config):
        super(Dreamer, self).__init__()
        self._config = config
        # this is update step
        self._wm = WorldModel(obs_space,config)
        self._task_behavior = ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)        
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: Random(config, act_space),
            plan2explore=lambda: Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)
    
    def __call__(self, obs, state=None):
      

        policy_output, state = self._policy(obs, state)

        return policy_output, state

    def _policy(self, obs, state):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        actor = self._task_behavior.actor(feat)
        action = actor.mode()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state
    
def get_model(path='agent.pt'):
    observation_space = gym.spaces.Dict(
          {
              # "image": gym.spaces.Box(0, 255, self._map_size, dtype=np.uint8),
              "is_first": gym.spaces.Box(0,1, shape=(1,), dtype=np.uint8),
              "is_last": gym.spaces.Box(0,1, shape=(1,), dtype=np.uint8),
              "is_terminal": gym.spaces.Box(0,1, shape=(1,), dtype=np.uint8),
              "lataccel": gym.spaces.Box(-np.inf, np.inf, shape=(2,),dtype=np.float32),#"log_player_pos": gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32
              "state": gym.spaces.Box(-np.inf, np.inf, shape=(3,),dtype=np.float32),
              "future_plan": gym.spaces.Box(-np.inf, np.inf, shape=(196,),dtype=np.float32),
              "image": gym.spaces.Box(0, 255, (64,64) + (3,), dtype=np.uint8),

          }
    )
    actions = gym.spaces.Box(-2.0, 2.0, (1,), dtype=np.float32)

    yml = yaml.YAML(typ="safe", pure=True)
    configs = yml.load(
        (pathlib.Path("dreamerv3-torch/configs.yaml").read_text())
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", "comma"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    config = parser.parse_args([])

    config.num_actions = actions.n if hasattr(actions, "n") else actions.shape[0]

    model = Dreamer(
        observation_space,
        actions,
        config
    ).to(config.device)

    model.requires_grad_(requires_grad=False)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["agent_state_dict"])
    tools.recursively_load_optim_state_dict(model, checkpoint["optims_state_dict"])

    return model

    