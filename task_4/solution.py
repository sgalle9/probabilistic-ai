import warnings
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from torch.distributions import Normal

from utils import ReplayBuffer, get_env, run_episode

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(NeuralNetwork, self).__init__()

        layers = [nn.Linear(input_dim, hidden_size)]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))

        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_size, output_dim)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            s = self.activation(layer(s))

        return self.output_layer(s)
    
class Actor:
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        self.network = NeuralNetwork(self.state_dim, self.action_dim * 2, self.hidden_size, self.hidden_layers, "relu")
        self.network.to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.actor_lr)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        mean, log_std = self.network(state).chunk(2, dim=-1)
        log_std = self.clamp_log_std(log_std)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.rsample()

        epsilon = 1e-6  # or some other small number
        log_prob = dist.log_prob(action) - torch.log(1 - torch.square(torch.tanh(action)) + epsilon).sum(dim=-1, keepdim=True)

        action = torch.tanh(action)         # action must be bounded by [-1, 1].

        assert action.shape == (state.shape[0], self.action_dim) and \
            log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
        return action, log_prob


class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: float, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        self.network = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers, "relu")
        self.network.to(self.device)

        self.target_network = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers, "relu")
        self.target_network.to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.critic_lr)



class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()

    def setup_agent(self):
        hidden_size = 256
        hidden_layers = 2
        actor_lr = 3e-3
        critic_lr = 3e-3
        alpha_lr = 3e-3
        alpha_init = 0.1

        self.tau = 0.005
        self.gamma = 0.99
        self.alpha = TrainableParameter(alpha_init, alpha_lr, True, self.device)
        self.target_entropy = - self.action_dim

        self.actor = Actor(hidden_size=hidden_size, hidden_layers=hidden_layers, actor_lr=actor_lr, state_dim=self.state_dim, action_dim=self.action_dim, device=self.device)

        self.critic1= Critic(hidden_size=hidden_size, hidden_layers=hidden_layers, critic_lr=critic_lr, state_dim=self.state_dim, action_dim=self.action_dim, device=self.device)

        self.critic2= Critic(hidden_size=hidden_size, hidden_layers=hidden_layers, critic_lr=critic_lr, state_dim=self.state_dim, action_dim=self.action_dim, device=self.device)
        


    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        s_tensor = torch.tensor(s).unsqueeze(0).to(self.device)

        if train:
            action, _ = self.actor.get_action_and_log_prob(s_tensor, deterministic=False)
        else:
            action, _ = self.actor.get_action_and_log_prob(s_tensor, deterministic=True)
        action = action.squeeze(0).cpu().detach().numpy()

        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch


        states = torch.tensor(s_batch, device=self.device)
        actions = torch.tensor(a_batch, device=self.device)
        rewards = torch.tensor(r_batch, device=self.device)
        next_states = torch.tensor(s_prime_batch, device=self.device)

        with torch.no_grad():
            action_next, log_pi_next = self.actor.get_action_and_log_prob(next_states, False)
            q1 = self.critic1.target_network(torch.cat([next_states, action_next], 1))
            q2 = self.critic2.target_network(torch.cat([next_states, action_next], 1))
            q_target = torch.min(q1, q2)

            v_target = q_target - self.alpha.get_param() * log_pi_next
        
        q1_pred = self.critic1.network(torch.cat([states, actions], 1))
        q1_loss = torch.nn.functional.mse_loss(q1_pred, (rewards + self.gamma * v_target))

        q2_pred = self.critic2.network(torch.cat([states, actions], 1))
        q2_loss = torch.nn.functional.mse_loss(q2_pred, (rewards + self.gamma * v_target))

        self.run_gradient_update_step(self.critic1, q1_loss)

        self.run_gradient_update_step(self.critic2, q2_loss)


        for param in self.critic1.network.parameters():
            param.requires_grad = False

        for param in self.critic2.network.parameters():
            param.requires_grad = False

        actions_pred, log_pi_pred = self.actor.get_action_and_log_prob(states, False)
        q1_pred = self.critic1.network(torch.cat([states, actions_pred], 1))
        q2_pred = self.critic2.network(torch.cat([states, actions_pred], 1))
        q_pred = torch.min(q1_pred, q2_pred)

        policy_loss = (self.alpha.get_param() * log_pi_pred - q_pred).mean()
        self.run_gradient_update_step(self.actor, policy_loss)

        # Update alpha.
        with torch.no_grad():
            actions_pred, log_pi_pred = self.actor.get_action_and_log_prob(states, False)
            # _, log_std = self.actor.network(states).chunk(2, dim=-1)
            # var = torch.square(torch.exp(log_std))
            # # Compute the entropy of the gaussian.
            # entropy = torch.log(2 * torch.pi * torch.e * var)
        
        alpha_loss = (- self.alpha.get_param() * (log_pi_pred + self.target_entropy)).mean()

        self.alpha.optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha.optimizer.step()


        for param in self.critic1.network.parameters():
            param.requires_grad = True
            
        for param in self.critic2.network.parameters():
            param.requires_grad = True


        self.critic_target_update(self.critic1.network, self.critic1.target_network, self.tau, True)

        self.critic_target_update(self.critic2.network, self.critic2.target_network, self.tau, True)


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = True
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
