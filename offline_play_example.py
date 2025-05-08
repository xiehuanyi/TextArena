import textarena as ta

# Initialize agents
agents = {
    0: ta.agents.OpenAIAgent(model_name="qwen-2.5", api_key='', base_url='http://localhost:8000/v1'),
    1: ta.agents.OpenAIAgent(model_name="qwen-2.5", api_key='', base_url='http://localhost:8000/v1'),
}

# Initialize environment from subset and wrap it
env = ta.make(env_id="SpellingBee-v0")
env = ta.wrappers.LLMObservationWrapper(env=env)
# Optional render wrapper 
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "GPT-4o-mini", 1: "claude-3.5-haiku"},
)

env.reset(num_players=len(agents))
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, info = env.step(action=action)
rewards = env.close()