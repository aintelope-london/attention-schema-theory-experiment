import datetime
import os
import torch
from aintelope.agents.model.model import Model
from aintelope.agents.model.dl_utils import select_checkpoint
from aintelope.agents.model.memory import ReplayMemory


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using GPU: " + str(self.device != torch.device("cpu")))
    
    def add_agent(self, agent_id, env_manifesto):
        self.models[agent_id] = Model(agent_id, env_manifesto, self.cfg, self.device)
    
    def reset_agent(self, agent_id):
        self.models[agent_id].memory = ReplayMemory(self.cfg.dl_params.batch_size)
    
    def get_action(self, agent_id, observation):
        return self.models[agent_id].get_action(observation)
    
    def update(self, agent_id, state, action, reward, done, next_state):
        self.models[agent_id].update(state, action, reward, done, next_state)
    
    def save_models(self, episode):
        path = os.path.normpath(self.cfg.addresses.pipeline_dir + "checkpoints/")
        os.makedirs(path, exist_ok=True)
        
        for agent_id, model in self.models.items():
            for role, component in model.connectome.items():
                if isinstance(component, torch.nn.Module):
                    checkpoint_filename = f"{self.cfg.addresses.experiment_name}_{episode}_{agent_id}_{role}"
                    filename = os.path.join(path, checkpoint_filename + "-" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
                    checkpoint_data = component.checkpoint_data()
                    checkpoint_data["epoch"] = episode
                    torch.save(checkpoint_data, filename)