import gymnasium as gym
from .g1_cat_env_cfg import G1CatEnvCfg

gym.register(
    id="Isaac-G1-Cat-Direct-v0",
    entry_point="whole_body_control.tasks.g1_cat.g1_cat_env:G1CatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1CatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1CatPPORunnerCfg",
    },
)
