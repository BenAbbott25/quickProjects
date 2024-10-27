import torch
import torch.nn.functional as F
import torch.optim as optim

def train_model(num_training_steps, batch_size, discount, tau, device, replay_buffer, actor, critic, target_actor, target_critic, actor_optimizer, critic_optimizer):
    for _ in range(num_training_steps):
        # Sample replay buffer
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(batch_states).to(device)
        next_state_batch = torch.FloatTensor(batch_next_states).to(device)
        action_batch = torch.FloatTensor(batch_actions).to(device)
        reward_batch = torch.FloatTensor(batch_rewards).to(device)
        done_batch = torch.FloatTensor(batch_dones).to(device)

        # Train critic
        target_actions = target_actor(next_state_batch)
        target_next_state_values = target_critic(next_state_batch, target_actions).detach()
        target_next_state_values = target_next_state_values.squeeze(1)
        expected_values = reward_batch + ((1 - done_batch) * discount * target_next_state_values).detach()
        action_batch = action_batch.squeeze(1)
        critic_values = critic(state_batch, action_batch).squeeze(1)

        critic_loss = F.mse_loss(critic_values.squeeze(-1), expected_values)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Train actor
        actor_loss = -critic(state_batch, actor(state_batch)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(target_actor.parameters(), actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(target_critic.parameters(), critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
