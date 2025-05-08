import textarena as ta 
import sys

hf_cfg = {"temperature": 0.7}
# model_ids = [
#     "Qwen/Qwen2.5-7B-Instruct",
#     "Qwen/Qwen2.5-14B-Instruct",
#     "Qwen/Qwen2.5-32B-Instruct",
#     "Qwen/Qwen2.5-72B-Instruct",
#     "meta-llama/Meta-Llama-3-8B-Instruct",
#     "meta-llama/Meta-Llama-3-70B-Instruct",
#     "microsoft/phi-3-mini-128k-instruct",
#     "microsoft/phi-3-small-128k-instruct",
#     "microsoft/phi-3-medium-128k-instruct",
#     "microsoft/phi-4-mini-instruct",
#     "mistralai/Mistral-7B-Instruct-v0.3",
#     "deepseek-ai/deepseek-v2-lite-chat"
# # ]
# model_ids = ["Qwen/Qwen2.5-7B-Instruct", 
#              "meta-llama/Meta-Llama-3-8B-Instruct", 
#              "microsoft/phi-3-mini-128k-instruct", 
#              "mistralai/Mistral-7B-Instruct-v0.3", 
#              "microsoft/phi-4-mini-instruct"]
fout = open("Eval-results.txt", 'w')
agents = {
    0: ta.agents.OpenAIAgent(model_name="", api_key="", base_url="http://localhost:8000/v1"),
    1: ta.agents.OpenAIAgent(model_name="", api_key="", base_url="http://localhost:8001/v1"),
}
for env_id in ['TicTacToe-v0', 'DontSayIt-v0', 'Poker-v0', 'Snake-v0']:
    env = ta.make(env_id=env_id)
    env = ta.wrappers.LLMObservationWrapper(env=env)
    env = ta.wrappers.SimpleRenderWrapper(env=env, render_mode="standard")
    winners = []

    for _ in range(3):
        env.reset(num_players=len(agents))
        done = False 
        while not done:
            player_id, observation = env.get_observation()
            action = agents[player_id](observation)
            done, info = env.step(action=action)
        rewards = env.close()
        winners.append(0 if rewards[0] > rewards[1] else 1)
    print(f"Winner: {winners}")
    fout.write(f"{env_id}\n")
    fout.write(f"{sys.argv[1]} vs DeepSeek-R1-Distill-Qwen-32B: {winners}\n")

